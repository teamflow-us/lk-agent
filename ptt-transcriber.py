import asyncio
import json
import logging
import os
import sys

from livekit import rtc
from livekit.agents import (
    AgentServer,
    AutoSubscribe,
    JobContext,
    cli,
    stt,
    utils,
)
from livekit.plugins import deepgram
import redis.asyncio as redis

# Add parent directory for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import REDIS_URL, AUDIO_STREAM, RESULT_CHANNEL_PREFIX
from shared.models import TranscriptionRequest, TranscriptionResult

# Standard LiveKit transcription topic
TOPIC_TRANSCRIPTION = "lk.transcription"

logger = logging.getLogger("ptt-transcriber")


class RedisClient:
    """Simple Redis client for the PTT agent."""

    def __init__(self, url: str = REDIS_URL):
        self._url = url
        self._redis: redis.Redis | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Redis. Returns True if successful."""
        try:
            self._redis = redis.from_url(self._url, decode_responses=False)
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self._url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        self._connected = False

    async def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._redis or not self._connected:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            self._connected = False
            return False

    async def send_audio(self, request: TranscriptionRequest) -> bool:
        """Send a transcription request to the audio stream."""
        if not await self.is_connected():
            return False
        try:
            await self._redis.xadd(AUDIO_STREAM, {"data": request.to_json()})
            logger.debug(f"Sent audio request {request.request_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")
            self._connected = False
            return False

    async def subscribe_to_room(self, room_id: str):
        """Subscribe to transcription results for a room."""
        channel = f"{RESULT_CHANNEL_PREFIX}{room_id}"
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        logger.info(f"Subscribed to {channel}")

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        result = TranscriptionResult.from_json(data)
                        yield result
                    except Exception as e:
                        logger.error(f"Error parsing result: {e}")
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()


class ParticipantPTTHandler:
    """Manages PTT state and audio collection for a single participant."""

    def __init__(
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.RemoteAudioTrack,
        room: rtc.Room,
        room_id: str,
        redis_client: RedisClient,
        fallback_stt: deepgram.STT,
    ):
        self.participant = participant
        self.identity = participant.identity
        self.name = participant.name or participant.identity
        self._track = track
        self._room = room
        self._room_id = room_id
        self._redis = redis_client
        self._fallback_stt = fallback_stt

        self._ptt_active = False
        self._audio_buffer: list[bytes] = []
        self._audio_stream: rtc.AudioStream | None = None
        self._audio_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._use_fallback = False
        self._sample_rate = 48000

    async def on_track_unmuted(self):
        """PTT pressed - start collecting audio."""
        async with self._lock:
            if self._ptt_active:
                return

            self._ptt_active = True
            self._audio_buffer = []
            logger.info(f"PTT START: {self.identity}")

            try:
                self._audio_stream = rtc.AudioStream(self._track)
                self._audio_task = asyncio.create_task(self._collect_audio())
            except Exception as e:
                logger.error(f"Failed to start PTT for {self.identity}: {e}")
                self._ptt_active = False
                await self._cleanup()

    async def on_track_muted(self):
        """PTT released - send audio for transcription."""
        async with self._lock:
            if not self._ptt_active:
                return

            self._ptt_active = False
            logger.info(f"PTT END: {self.identity}")

            # Cancel audio collection
            if self._audio_task:
                self._audio_task.cancel()
                try:
                    await self._audio_task
                except asyncio.CancelledError:
                    pass

            # Close audio stream
            if self._audio_stream:
                try:
                    await self._audio_stream.aclose()
                except Exception as e:
                    logger.warning(f"Error closing audio stream: {e}")

            # Get collected audio
            audio_data = b"".join(self._audio_buffer)
            self._audio_buffer = []

            if not audio_data:
                logger.info(f"No audio collected for {self.identity}")
                await self._cleanup()
                return

            # Try to send to transcription service
            if not self._use_fallback and await self._redis.is_connected():
                try:
                    request = TranscriptionRequest.create(
                        user_id=self.identity,
                        user_name=self.name,
                        room_id=self._room_id,
                        audio_data=audio_data,
                        sample_rate=self._sample_rate,
                    )
                    success = await self._redis.send_audio(request)
                    if success:
                        logger.debug(f"Sent audio to transcription service for {self.identity}")
                        await self._cleanup()
                        return
                except Exception as e:
                    logger.error(f"Failed to send to service: {e}")

            # Fallback: Direct Deepgram transcription
            self._use_fallback = True
            logger.info(f"Using fallback transcription for {self.identity}")
            await self._transcribe_fallback(audio_data)
            await self._cleanup()

    async def _collect_audio(self):
        """Collect audio frames into buffer."""
        try:
            async for audio_event in self._audio_stream:
                if self._ptt_active:
                    frame = audio_event.frame
                    self._sample_rate = frame.sample_rate
                    self._audio_buffer.append(frame.data.tobytes())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Audio collection error for {self.identity}: {e}")

    async def _transcribe_fallback(self, audio_data: bytes):
        """Fallback: transcribe directly using Deepgram."""
        try:
            stream = self._fallback_stt.stream()

            # Create audio frame
            num_samples = len(audio_data) // 2
            frame = rtc.AudioFrame(
                data=audio_data,
                sample_rate=self._sample_rate,
                num_channels=1,
                samples_per_channel=num_samples,
            )

            stream.push_frame(frame)
            stream.flush()
            stream.end_input()

            # Collect transcription
            transcript_parts = []
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    if event.alternatives and event.alternatives[0].text:
                        transcript_parts.append(event.alternatives[0].text)

            await stream.aclose()

            transcript = " ".join(transcript_parts).strip()
            if transcript:
                logger.info(f"{self.name} ({self.identity}) -> {transcript}")
                await self._broadcast_to_room(transcript)

        except Exception as e:
            logger.error(f"Fallback transcription error: {e}")

    async def _broadcast_to_room(self, text: str):
        """Broadcast transcription to the room."""
        try:
            payload = json.dumps({
                "participant_identity": self.identity,
                "participant_name": self.name,
                "text": text,
                "is_final": True,
            })
            await self._room.local_participant.publish_data(
                payload=payload,
                topic=TOPIC_TRANSCRIPTION,
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to broadcast: {e}")

    async def _cleanup(self):
        """Clean up resources."""
        self._audio_stream = None
        self._audio_task = None
        self._audio_buffer = []

    async def aclose(self):
        """Clean up all resources when participant disconnects."""
        async with self._lock:
            self._ptt_active = False
            if self._audio_task:
                self._audio_task.cancel()
                try:
                    await self._audio_task
                except asyncio.CancelledError:
                    pass
            if self._audio_stream:
                try:
                    await self._audio_stream.aclose()
                except Exception:
                    pass
            await self._cleanup()


class MultiUserPTTTranscriber:
    """Manages PTT transcription for multiple participants."""

    def __init__(self, ctx: JobContext, room_id: str):
        self.ctx = ctx
        self._room_id = room_id
        self._handlers: dict[str, ParticipantPTTHandler] = {}
        self._redis = RedisClient()
        self._fallback_stt = deepgram.STT()
        self._tasks: set[asyncio.Task] = set()
        self._subscription_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None

    async def start(self):
        """Start the transcriber."""
        # Try to connect to Redis
        await self._redis.connect()

        # Start result subscription if Redis is available
        if await self._redis.is_connected():
            self._subscription_task = asyncio.create_task(self._subscribe_to_results())
            self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Register room event handlers
        room = self.ctx.room
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_muted", self._on_track_muted)
        room.on("track_unmuted", self._on_track_unmuted)
        room.on("participant_disconnected", self._on_participant_disconnected)

    async def aclose(self):
        """Clean up all handlers and tasks."""
        # Cancel subscription task
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        await utils.aio.cancel_and_wait(*self._tasks)

        # Close all participant handlers
        await asyncio.gather(
            *[handler.aclose() for handler in self._handlers.values()],
            return_exceptions=True
        )
        self._handlers.clear()

        # Close Redis
        await self._redis.close()

        # Unregister event handlers
        room = self.ctx.room
        room.off("track_subscribed", self._on_track_subscribed)
        room.off("track_muted", self._on_track_muted)
        room.off("track_unmuted", self._on_track_unmuted)
        room.off("participant_disconnected", self._on_participant_disconnected)

    async def _subscribe_to_results(self):
        """Listen for transcription results from the service."""
        try:
            async for result in self._redis.subscribe_to_room(self._room_id):
                await self._broadcast_result(result)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Result subscription error: {e}")

    async def _broadcast_result(self, result: TranscriptionResult):
        """Broadcast a transcription result to the room."""
        try:
            payload = result.to_broadcast_payload()
            await self.ctx.room.local_participant.publish_data(
                payload=payload,
                topic=TOPIC_TRANSCRIPTION,
                reliable=True,
            )
            logger.info(f"{result.user_name} ({result.user_id}) -> {result.text}")
        except Exception as e:
            logger.error(f"Failed to broadcast result: {e}")

    async def _health_check_loop(self):
        """Periodically check if Redis is back online."""
        while True:
            try:
                await asyncio.sleep(30)
                if await self._redis.is_connected():
                    # Redis is healthy, ensure handlers use service
                    for handler in self._handlers.values():
                        handler._use_fallback = False
                else:
                    # Try to reconnect
                    if await self._redis.connect():
                        logger.info("Redis reconnected")
                        for handler in self._handlers.values():
                            handler._use_fallback = False
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Create handler when audio track is subscribed."""
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        if participant.identity in self._handlers:
            return

        logger.info(f"Audio track subscribed for {participant.identity}")

        handler = ParticipantPTTHandler(
            participant=participant,
            track=track,
            room=self.ctx.room,
            room_id=self._room_id,
            redis_client=self._redis,
            fallback_stt=self._fallback_stt,
        )
        self._handlers[participant.identity] = handler

    def _on_track_muted(
        self,
        participant: rtc.RemoteParticipant,
        publication: rtc.RemoteTrackPublication,
    ):
        """Route mute event to participant handler."""
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        logger.info(f"Track muted event for {participant.identity}")
        handler = self._handlers.get(participant.identity)
        if handler:
            task = asyncio.create_task(handler.on_track_muted())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _on_track_unmuted(
        self,
        participant: rtc.RemoteParticipant,
        publication: rtc.RemoteTrackPublication,
    ):
        """Route unmute event to participant handler."""
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        logger.info(f"Track unmuted event for {participant.identity}")
        handler = self._handlers.get(participant.identity)
        if handler:
            task = asyncio.create_task(handler.on_track_unmuted())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Clean up handler when participant disconnects."""
        handler = self._handlers.pop(participant.identity, None)
        if handler:
            logger.info(f"Closing handler for {participant.identity}")
            task = asyncio.create_task(handler.aclose())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Get room ID for this session
    room_id = ctx.room.name

    transcriber = MultiUserPTTTranscriber(ctx, room_id)
    await transcriber.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Handle existing participants' tracks
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track and publication.kind == rtc.TrackKind.KIND_AUDIO:
                transcriber._on_track_subscribed(
                    publication.track, publication, participant
                )

    async def cleanup():
        await transcriber.aclose()

    ctx.add_shutdown_callback(cleanup)


if __name__ == "__main__":
    cli.run_app(server)

import asyncio
import logging

import json

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

# Standard LiveKit transcription topic
TOPIC_TRANSCRIPTION = "lk.transcription"

logger = logging.getLogger("ptt-transcriber")


class ParticipantPTTHandler:
    """Manages PTT state and Deepgram streams for a single participant."""

    def __init__(
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.RemoteAudioTrack,
        stt_instance: deepgram.STT,
        room: rtc.Room,
    ):
        self.participant = participant
        self.identity = participant.identity
        self._track = track
        self._stt = stt_instance
        self._room = room

        self._ptt_active = False
        self._stream: stt.SpeechStream | None = None
        self._audio_stream: rtc.AudioStream | None = None
        self._audio_task: asyncio.Task | None = None
        self._stt_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def on_track_unmuted(self):
        """PTT pressed - start Deepgram stream."""
        async with self._lock:
            if self._ptt_active:
                return  # Already active (debounce)

            self._ptt_active = True
            logger.info(f"PTT START: {self.identity}")

            try:
                # Create new Deepgram stream (new WebSocket connection)
                self._stream = self._stt.stream()

                # Create AudioStream from track
                self._audio_stream = rtc.AudioStream(self._track)

                # Start async tasks for audio forwarding and STT consumption
                self._audio_task = asyncio.create_task(self._audio_loop())
                self._stt_task = asyncio.create_task(self._consume_stt_events())
            except Exception as e:
                logger.error(f"Failed to start PTT for {self.identity}: {e}")
                self._ptt_active = False
                await self._cleanup()

    async def on_track_muted(self):
        """PTT released - finalize transcription."""
        async with self._lock:
            if not self._ptt_active:
                return

            self._ptt_active = False
            logger.info(f"PTT END: {self.identity}")

            # Signal end of input to Deepgram
            if self._stream:
                try:
                    self._stream.flush()
                    self._stream.end_input()
                except Exception as e:
                    logger.error(f"Error finalizing stream for {self.identity}: {e}")

            # Wait for STT task to get final transcript
            if self._stt_task:
                try:
                    await asyncio.wait_for(self._stt_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"STT task timeout for {self.identity}")
                except Exception as e:
                    logger.error(f"STT task error for {self.identity}: {e}")

            # Cancel audio task
            if self._audio_task:
                self._audio_task.cancel()
                try:
                    await self._audio_task
                except asyncio.CancelledError:
                    pass

            await self._cleanup()

    async def _audio_loop(self):
        """Forward audio frames to STT stream."""
        try:
            async for audio_event in self._audio_stream:
                if self._stream and self._ptt_active:
                    self._stream.push_frame(audio_event.frame)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Audio loop error for {self.identity}: {e}")

    async def _consume_stt_events(self):
        """Consume STT events and send transcriptions to the room."""
        final_transcript_parts = []
        try:
            async for event in self._stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    if event.alternatives and event.alternatives[0].text:
                        final_transcript_parts.append(event.alternatives[0].text)
                elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    # Optionally log interim results for debugging
                    pass
        except Exception as e:
            logger.error(f"STT consumption error for {self.identity}: {e}")

        # Send the complete transcript to the room
        final_transcript = " ".join(final_transcript_parts).strip()
        if final_transcript:
            logger.info(f"{self.identity} -> {final_transcript}")
            await self._publish_transcription(final_transcript)

    async def _publish_transcription(self, text: str):
        """Publish transcription to the room's data channel."""
        try:
            # Create transcription payload
            payload = json.dumps({
                "participant_identity": self.identity,
                "text": text,
                "is_final": True,
            })

            # Publish to the room
            await self._room.local_participant.publish_data(
                payload=payload,
                topic=TOPIC_TRANSCRIPTION,
                reliable=True,
            )
            logger.debug(f"Published transcription for {self.identity}")
        except Exception as e:
            logger.error(f"Failed to publish transcription for {self.identity}: {e}")

    async def _cleanup(self):
        """Clean up resources."""
        self._stream = None
        self._audio_stream = None
        self._audio_task = None
        self._stt_task = None

    async def aclose(self):
        """Clean up all resources when participant disconnects."""
        async with self._lock:
            self._ptt_active = False

            # Cancel ongoing tasks
            if self._audio_task:
                self._audio_task.cancel()
            if self._stt_task:
                self._stt_task.cancel()

            # Close STT stream
            if self._stream:
                try:
                    await self._stream.aclose()
                except Exception:
                    pass

            # Wait for tasks to complete
            tasks = [t for t in [self._audio_task, self._stt_task] if t]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            await self._cleanup()


class MultiUserPTTTranscriber:
    """Manages PTT transcription for multiple participants."""

    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._handlers: dict[str, ParticipantPTTHandler] = {}
        self._stt = deepgram.STT()  # Shared STT instance (streams are independent)
        self._tasks: set[asyncio.Task] = set()

    def start(self):
        """Register room event handlers."""
        room = self.ctx.room
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_muted", self._on_track_muted)
        room.on("track_unmuted", self._on_track_unmuted)
        room.on("participant_disconnected", self._on_participant_disconnected)

    async def aclose(self):
        """Clean up all handlers and tasks."""
        await utils.aio.cancel_and_wait(*self._tasks)

        # Close all participant handlers
        await asyncio.gather(
            *[handler.aclose() for handler in self._handlers.values()],
            return_exceptions=True
        )
        self._handlers.clear()

        # Unregister event handlers
        room = self.ctx.room
        room.off("track_subscribed", self._on_track_subscribed)
        room.off("track_muted", self._on_track_muted)
        room.off("track_unmuted", self._on_track_unmuted)
        room.off("participant_disconnected", self._on_participant_disconnected)

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
            return  # Already have a handler

        logger.info(f"Audio track subscribed for {participant.identity} (muted: {publication.muted})")

        handler = ParticipantPTTHandler(
            participant=participant,
            track=track,
            stt_instance=self._stt,
            room=self.ctx.room,
        )
        self._handlers[participant.identity] = handler

        # Don't auto-start PTT on subscription - wait for explicit track_unmuted event
        # The track may briefly report as unmuted before the real mute state is known

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
    transcriber = MultiUserPTTTranscriber(ctx)
    transcriber.start()

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

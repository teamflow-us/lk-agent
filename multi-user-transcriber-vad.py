import asyncio
import logging
from typing import Optional

from livekit import rtc
from livekit.agents import (
    AgentServer,
    AutoSubscribe,
    JobContext,
    JobProcess,
    cli,
    utils,
)
from livekit.plugins import deepgram, silero

logger = logging.getLogger("transcriber")


class ParticipantTranscriber:
    """Transcribes a single participant, only connecting to Deepgram during speech."""

    def __init__(
        self,
        participant: rtc.RemoteParticipant,
        room: rtc.Room,
        vad: silero.VAD,
    ):
        self.participant = participant
        self.room = room
        self.vad = vad
        self._stt: Optional[deepgram.STT] = None
        self._audio_stream: Optional[rtc.AudioStream] = None
        self._tasks: set[asyncio.Task] = set()
        self._running = False

    async def start(self):
        """Start listening to the participant's audio."""
        self._running = True

        # Subscribe to track events
        self.participant.on("track_subscribed", self._on_track_subscribed)
        self.participant.on("track_unsubscribed", self._on_track_unsubscribed)

        # Handle already-subscribed tracks
        for pub in self.participant.track_publications.values():
            if pub.track and pub.kind == rtc.TrackKind.KIND_AUDIO:
                await self._start_audio_processing(pub.track)

    async def stop(self):
        """Stop transcribing this participant."""
        self._running = False
        self.participant.off("track_subscribed", self._on_track_subscribed)
        self.participant.off("track_unsubscribed", self._on_track_unsubscribed)

        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            task = asyncio.create_task(self._start_audio_processing(track))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        # Audio stream will be cleaned up automatically
        pass

    async def _start_audio_processing(self, track: rtc.Track):
        """Process audio with VAD, only transcribe when speech detected."""
        audio_stream = rtc.AudioStream(track)

        # Create VAD stream to detect speech
        vad_stream = self.vad.stream()

        async for audio_event in audio_stream:
            if not self._running:
                break

            # Push audio to VAD
            vad_stream.push_frame(audio_event.frame)

            # Check VAD events
            async for vad_event in vad_stream:
                if vad_event.type == silero.VADEventType.START_OF_SPEECH:
                    logger.info(f"Speech started: {self.participant.identity}")
                    # Create new STT instance for this utterance
                    self._stt = deepgram.STT(model="nova-2")

                elif vad_event.type == silero.VADEventType.END_OF_SPEECH:
                    logger.info(f"Speech ended: {self.participant.identity}")
                    if self._stt and vad_event.frames:
                        # Transcribe the complete utterance
                        await self._transcribe_utterance(vad_event.frames)
                    self._stt = None

        await vad_stream.aclose()

    async def _transcribe_utterance(self, frames: list[rtc.AudioFrame]):
        """Send complete utterance to Deepgram for transcription."""
        if not self._stt:
            return

        try:
            # Merge frames into single audio buffer
            merged = utils.audio.merge_frames(frames)

            # Use non-streaming recognize for the complete utterance
            result = await self._stt.recognize(
                buffer=merged,
                language="en",
            )

            if result.alternatives and result.alternatives[0].text:
                text = result.alternatives[0].text
                logger.info(f"{self.participant.identity} -> {text}")

                # Forward transcription via text stream
                await self._forward_transcription(text, frames[0] if frames else None)

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    async def _forward_transcription(self, text: str, frame: Optional[rtc.AudioFrame]):
        """Send transcription as a text stream to the room."""
        # Find the participant's audio track SID
        track_sid = None
        for pub in self.participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track:
                track_sid = pub.sid
                break

        # Create text stream with transcription attributes
        writer = self.room.local_participant.stream_text(
            topic="lk.transcription",
            attributes={
                "lk.transcription_final": "true",
                "lk.transcribed_track_id": track_sid or "",
                "lk.segment_id": f"SG_{utils.shortuuid()}",
            },
        )

        async with writer:
            await writer.write(text)


class MultiUserTranscriber:
    """Manages transcription for all participants in a room."""

    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._transcribers: dict[str, ParticipantTranscriber] = {}
        self._tasks: set[asyncio.Task] = set()

    def start(self):
        self.ctx.room.on("participant_connected", self._on_participant_connected)
        self.ctx.room.on("participant_disconnected", self._on_participant_disconnected)

    async def aclose(self):
        await utils.aio.cancel_and_wait(*self._tasks)

        for transcriber in self._transcribers.values():
            await transcriber.stop()
        self._transcribers.clear()

        self.ctx.room.off("participant_connected", self._on_participant_connected)
        self.ctx.room.off("participant_disconnected", self._on_participant_disconnected)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        if participant.identity in self._transcribers:
            return

        logger.info(f"Starting transcriber for {participant.identity}")
        transcriber = ParticipantTranscriber(
            participant=participant,
            room=self.ctx.room,
            vad=self.ctx.proc.userdata["vad"],
        )
        self._transcribers[participant.identity] = transcriber

        task = asyncio.create_task(transcriber.start())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        transcriber = self._transcribers.pop(participant.identity, None)
        if transcriber:
            logger.info(f"Stopping transcriber for {participant.identity}")
            task = asyncio.create_task(transcriber.stop())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    transcriber = MultiUserTranscriber(ctx)
    transcriber.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Handle existing participants
    for participant in ctx.room.remote_participants.values():
        transcriber._on_participant_connected(participant)

    ctx.add_shutdown_callback(transcriber.aclose)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

if __name__ == "__main__":
    cli.run_app(server)

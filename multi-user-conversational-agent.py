import asyncio
import logging
import os
import time


from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    cli,
    llm,
    room_io,
    utils,
)
from livekit.plugins import deepgram, silero, openai


logger = logging.getLogger("conversational-agent")


# This agent demonstrates how to have voice conversations with multiple remote participants.
# It creates agent sessions for each participant and responds with text-to-speech.
class ConversationalAgent(Agent):
    def __init__(self, *, participant_identity: str):
        # Load configuration from environment variables with defaults
        tts_voice = os.getenv("AGENT_TTS_VOICE", "nova")
        tts_model = os.getenv("AGENT_TTS_MODEL", "tts-1")
        tts_speed = float(os.getenv("AGENT_TTS_SPEED", "1.0"))
        llm_model = os.getenv("AGENT_LLM_MODEL", "gpt-3.5-turbo")
        instructions = os.getenv(
            "AGENT_INSTRUCTIONS", 
            """You are a helpful StageLink assistant. 
            - Be concise and professional
            - Focus on audio/video collaboration topics
            - Keep responses under 2 sentences"""
        )
        
        super().__init__(
            instructions=instructions,
            llm=openai.LLM(model=llm_model),
            tts=openai.TTS(
                voice=tts_voice,
                model=tts_model,
                speed=tts_speed
            ),
            stt=deepgram.STT(),
        )
        self.participant_identity = participant_identity
        # Add custom context per participant
        self.custom_context = {
            "participant_id": participant_identity,
            "session_start": time.time(),
            "conversation_history": []
        }

    async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        user_transcript = new_message.text_content
        
        # Add custom context to chat context for personalized responses
        chat_ctx.append(
            text=f"Current participant: {self.participant_identity}. Session duration: {time.time() - self.custom_context['session_start']:.0f} seconds.",
            role="system"
        )
        
        logger.info(f"{self.participant_identity} -> {user_transcript}")
        
        # The agent will automatically respond using the LLM and TTS with custom context


class MultiUserConversationalAgent:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._sessions: dict[str, AgentSession] = {}
        self._tasks: set[asyncio.Task] = set()

    def start(self):
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)

    async def aclose(self):
        await utils.aio.cancel_and_wait(*self._tasks)

        await asyncio.gather(*[self._close_session(session) for session in self._sessions.values()])

        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        if participant.identity in self._sessions:
            return

        logger.info(f"starting session for {participant.identity}")
        task = asyncio.create_task(self._start_session(participant))
        self._tasks.add(task)

        def on_task_done(task: asyncio.Task):
            try:
                self._sessions[participant.identity] = task.result()
            finally:
                self._tasks.discard(task)

        task.add_done_callback(on_task_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        if (session := self._sessions.pop(participant.identity)) is None:
            return

        logger.info(f"closing session for {participant.identity}")
        task = asyncio.create_task(self._close_session(session))
        self._tasks.add(task)
        task.add_done_callback(lambda _: self._tasks.discard(task))

    async def _start_session(self, participant: rtc.RemoteParticipant) -> AgentSession:
        if participant.identity in self._sessions:
            return self._sessions[participant.identity]

        session = AgentSession(
            vad=self.ctx.proc.userdata["vad"],
        )
        await session.start(
            agent=ConversationalAgent(
                participant_identity=participant.identity,
            ),
            room=self.ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=True,
                text_output=True,
                audio_output=True,  # Enable audio output for TTS responses
                participant_identity=participant.identity,
                # text input is not supported for multiple room participants
                # if needed, register the text stream handler by yourself
                # and route the text to different sessions based on the participant identity
                text_input=False,
            ),
        )
        return session

    async def _close_session(self, sess: AgentSession) -> None:
        await sess.drain()
        await sess.aclose()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    conversational_agent = MultiUserConversationalAgent(ctx)
    conversational_agent.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    for participant in ctx.room.remote_participants.values():
        # handle all existing participants
        conversational_agent.on_participant_connected(participant)

    async def cleanup():
        await conversational_agent.aclose()

    ctx.add_shutdown_callback(cleanup)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

if __name__ == "__main__":
    cli.run_app(server)
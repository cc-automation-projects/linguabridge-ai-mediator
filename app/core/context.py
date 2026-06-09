from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
channel_var: ContextVar[str] = ContextVar("channel", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")


def set_trace_id(trace_id: str) -> None:
    trace_id_var.set(trace_id)


def set_channel(channel: str) -> None:
    channel_var.set(channel)


def set_user_id(user_id: str) -> None:
    user_id_var.set(user_id)

from __future__ import annotations

from typing import Any, TYPE_CHECKING


def _load_neonize_symbols():
    restore_validate = None
    runtime_version = None

    try:
        import google.protobuf
        from google.protobuf import runtime_version

        major_version = int(str(google.protobuf.__version__).split(".", 1)[0])
        if major_version < 7:
            restore_validate = runtime_version.ValidateProtobufRuntimeVersion
            runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    except Exception:
        restore_validate = None

    try:
        from neonize import NewClient
        from neonize.proto import Neonize_pb2 as neonize_proto
        from neonize.proto.Neonize_pb2 import JID, Message as MessageEvent
        from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import ContextInfo, Message as WaMessage
        from neonize.utils import Jid2String, build_jid
        from neonize.utils.enum import ChatPresence, ChatPresenceMedia, ParticipantChange

        return (
            NewClient,
            neonize_proto,
            JID,
            MessageEvent,
            ContextInfo,
            WaMessage,
            Jid2String,
            build_jid,
            ChatPresence,
            ChatPresenceMedia,
            ParticipantChange,
        )
    finally:
        if restore_validate is not None and runtime_version is not None:
            runtime_version.ValidateProtobufRuntimeVersion = restore_validate


(
    NewClient,
    neonize_proto,
    JID,
    MessageEvent,
    ContextInfo,
    WaMessage,
    Jid2String,
    build_jid,
    ChatPresence,
    ChatPresenceMedia,
    ParticipantChange,
) = _load_neonize_symbols()

if TYPE_CHECKING:
    from neonize.proto.Neonize_pb2 import JID as JIDType, Message as MessageEventType, SendResponse as SendResponseType
    from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import ContextInfo as ContextInfoType, Message as WaMessageType
else:
    # Runtime aliases must be concrete classes (not typing.Any), because Neonize
    # uses these in decorators and isinstance-style checks.
    JIDType = JID
    MessageEventType = MessageEvent
    SendResponseType = Any
    ContextInfoType = ContextInfo
    WaMessageType = WaMessage

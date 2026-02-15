import dataclasses as dc


@dc.dataclass
class FileInput:
    path: str


@dc.dataclass
class ZMQInput:
    address: str


@dc.dataclass
class MofkaInput:
    group_file: str
    topic_name: str

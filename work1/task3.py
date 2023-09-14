from enum import Enum
from typing import Iterable

OperationId = str
OperationName = str


class OperationStatus(str, Enum):
    OK = "OK"
    ACCESS_DENIED = "Access denied"


class FileManager:
    OPERATION_NAME_TO_ID = {
        'read': 'r',
        'write': 'w',
        'execute': 'x',
    }

    def __init__(self):
        self.filename_to_operation_ids: dict[str, set[OperationId]] = {}

    def add_file(self, filename: str, operation_ids: Iterable[OperationId]):
        self.filename_to_operation_ids[filename] = set(operation_ids)

    def _check_file_for_operation(self, filename: str, operation_name: OperationName) -> OperationStatus:
        operation_id = self.OPERATION_NAME_TO_ID[operation_name]

        if (
            filename in self.filename_to_operation_ids
            and operation_id in self.filename_to_operation_ids[filename]
        ):
            return OperationStatus.OK

        return OperationStatus.ACCESS_DENIED

    def check_file_for_operation(self, filename: str, operation_name: OperationName) -> str:
        status = self._check_file_for_operation(filename=filename, operation_name=operation_name)
        return status.value


def main():
    n = int(input())
    file_manager = FileManager()

    for _ in range(n):
        filename, *operation_ids = input().split()
        file_manager.add_file(
            filename=filename,
            operation_ids=operation_ids,
        )

    m = int(input())

    for _ in range(m):
        operation_name, filename = input().split()
        answer = file_manager.check_file_for_operation(
            filename=filename,
            operation_name=operation_name,
        )

        print("result ->", answer)


if __name__ == "__main__":
    main()

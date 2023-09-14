from enum import Enum


class RegistrationStatus(str, Enum):
    OK = "OK"
    ALREADY_EXISTS = "ALREADY_EXISTS"


class UserManager:
    def __init__(self):
        self.username_to_count: dict[str, int] = {}

    def _validate_registration(self, username: str) -> RegistrationStatus:
        if username in self.username_to_count:
            return RegistrationStatus.ALREADY_EXISTS
        return RegistrationStatus.OK

    def _suggest_username(self, username: str) -> str:
        new_username = f"{username}{self.username_to_count[username]}"
        self.username_to_count[username] += 1
        return new_username

    def _register(self, username: str):
        self.username_to_count[username] = 1

    def maybe_register(self, username: str) -> str:
        status = self._validate_registration(username=username)

        if status == RegistrationStatus.OK:
            self._register(username=username)
            return status.value

        return self._suggest_username(username=username)


def main():
    n = int(input())
    user_manager = UserManager()

    for _ in range(n):
        username = input()
        answer = user_manager.maybe_register(username=username)

        print("result ->", answer)


if __name__ == "__main__":
    main()

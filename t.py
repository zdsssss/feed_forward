class a:
    def __init__(self) -> None:
        print("s")
    def __call__(self, a):
        print(a)
A = a()
A("asdf")
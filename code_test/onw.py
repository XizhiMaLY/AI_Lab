# define a parent class and a child class for practice purposes
class Animal:
    _ani = '666'
    print(_ani)
    def __init__(self, name) -> None:
        self.name = name

    def speak(self):
        print(f"{self.name} says something")
        pass

class Dog(Animal):
    _anii = '777'
    def __init__(self) -> None:
        super().__init__("Dog")
        print(self._ani)

    def speak(self):
        print(f"{self.name} says woof")
d = Dog()
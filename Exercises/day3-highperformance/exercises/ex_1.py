#### a. Create a "Person" class which takes firstname and lastname as arguments to the constructor 
# (```___init___```) and define a method that returns the full name of the person as a combined string.


class Person( ):

    def __init__(self, name, surname):
        self.name=name
        self.surname=surname

    def get_name_surname(self):
        return(f"{self.name} {self.surname}")
    


#### b. Create a "Student" class which inherits from the "Person" class, takes the subject area as an
#  additional argument to the constructor and define a method that prints the full name and the subject area of the student.

class Student(Person):

    def __init__(self, name, surname, subject):

        Person.__init__(self, name, surname)


        self.subject = subject


    def get_info(self):

        name_surname= Person.get_name_surname(self)

        return (f"{name_surname} {self.subject}")
    

class Teacher(Person):

    def __init__(self, name, surname, subject):

        Person.__init__(self, name, surname)


        self.subject = subject


    def get_info(self):

        name_surname= Person.get_name_surname(self)

        return (f"{name_surname} {self.subject}")



if __name__== "__main__":

    person=Person("Miriana", "Passarotto")
    name_surname=person.get_name_surname()
    print(name_surname)   


    student= Student("Miriana", "Passarotto", "Scienze")

    print(student.get_info())

    teacher=Teacher("Benedikt", "Daurer", "physics")

    print(teacher.get_info())
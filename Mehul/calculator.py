# # def my_function(x):
# #     print(x * 2)

# # def add_numbers(a, b):
# #     print(a + b)


# # my_function(5)
# # add_numbers(3, 4)


# class Student: #class name , awlays start with capital letter 
#     college  = "NSUT"
#     def __init__(self, name, roll_no): # constructor, it takes two parameter names : name, roll_no , always remeber funtion start with self 
#         self.name = name
#         self.roll_no = roll_no
#     def display(self):
#         print(self.name, self.roll_no)


# # obj = Student(name="Mehul", roll_no=123)
# # obj.display()

# class Calculator:  
#     def add(self, a, b):
#         return a + b

#     def subtract(self, a, b):
#         return a - b

#     def multiply(self, a, b):
#         return a * b

#     def divide(self, a, b):
#         if b != 0:
#             return a / b
#         else:
#             return "Error: Division by zero"

# obj = Calculator()







# # write student class , take first name, last name, and display full name 

# class Student: # class is keyword, Student is class name, always start with capital letter
#     def display_full_name(self, first_name, last_name): # self is a keyword which we use for methods inside the class 
#         print(first_name + " " + last_name)

#     def display_college_name(self): # self is a keyword which we use for methods inside the class 
#         print("NSUT")

# class_obj = Student() # create object of class Student, to access the methods inside the class 

# class_obj.display_full_name("Mehul", "Nigam")
# class_obj.display_college_name()






# class NSUTStudent:
#     college_name = "NSUT"  # class variable

#     def __init__(self, first_name, last_name):  # constructor
#         self.first_name = first_name  # instance variable
#         self.last_name = last_name  # instance variable

#     def display_full_name(self):  # method to display full name
#         print(self.first_name + " "+ self.last_name)

#     def display_college_name(self):  # method to display college name
#         print(f"College Name: {NSUTStudent.college_name}")



    



# class Quadrilateral:
#     def __init__(self, a, b, c, d):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.d = d
         

#     def perimeter(self):
#         return self.a + self.b + self.c + self.d

# obj = Quadrilateral(a=10,b=20, c=30, d =40)
# print(obj.perimeter())

# class Reactangle(Quadrilateral):
#     def __init__(self, length, breadth):
#         super().__init__(length, breadth, length, breadth)

# # Inheritance means a child class gets the data and methods of a parent class for free, so you don't rewrite the same code.

# obj = Reactangle(100,200)
# print(obj.perimeter())

# class Square(Reactangle):
#     def __init__(self, side):
#         super().__init__(side, side) # super refer to parent class


# obj1 = Square(25)
# print(obj1.perimeter())


# """
# Term	                                         Meaning
# Parent / Base class	                         The class being inherited from (Quadrilateral)
# Child / Derived class	                     The class that inherits (Rectangle, Square)
# super()	                                          Refers to the parent class
# Inheritance	                                 Child reuses the parent's methods and attributes
# Multilevel inheritance	                     A chain: grandparent → parent → child (Quadrilateral → Rectangle → Square)

# """










class Quad:
    def __init__(self, a,b,c,d): # Costructore  
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def area(self):
        print(self.a * self.b  )

obj1 = Quad(1,2,3,4)
obj1.area()

class React(Quad):
    def __init__(self, x,y): # init is a constrcutore
        super().__init__(x,y,x,y)

obj2 = React(20,30)
obj2.area()

class Square(Quad):
    def __init__(self, x):
        super().__init__(x,x,x,x)

obj3 = Square(10)
obj3.area()
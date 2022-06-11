from tkinter import *
from tkinter import messagebox
import draw_iris_dataset
import main
import numpy as np



Master=Tk()
Master.geometry("900x300")







############################ Draw the Data  ##############################

Label(Master,text="Any Two Features Do you need to Draw ??",font='helvetica 12 bold ').grid(row=0,column=0)
select=IntVar()
def Rclicked(value):
    select.set(value)

Radiobutton(Master,text="X1 and X2",variable=select,value=1,command=lambda :Rclicked(1)).grid(row=1,column=0)
Radiobutton(Master,text="X1 and X3",variable=select,value=2,command=lambda :Rclicked(2)).grid(row=2,column=0)
Radiobutton(Master,text="X1 and X4",variable=select,value=3,command=lambda :Rclicked(3)).grid(row=3,column=0)
Radiobutton(Master,text="X2 and X3",variable=select,value=4,command=lambda :Rclicked(4)).grid(row=4,column=0)
Radiobutton(Master,text="X2 and X4",variable=select,value=5,command=lambda :Rclicked(5)).grid(row=5,column=0)
Radiobutton(Master,text="X3 and X4",variable=select,value=6,command=lambda :Rclicked(6)).grid(row=6,column=0)


def draw_button(value):
    if(value==1):
        draw_iris_dataset.draw_x1_x2()
        Label(Master,text="X1 and X2 are descriminative between C1 and C2 && C3 only",bg='red',font='helvetica 8 bold ').place(x=20,y=180)
    elif(value==2):
        draw_iris_dataset.draw_x1_x3()
        Label(Master, text="X1 and X2 are descriminative between C1 and C2 && C3           ", bg='red',
              font='helvetica 8 bold ').place(x=20, y=180)
        Label(Master,
              text="but we can seperate between C2 and C3 hardly          ",
              bg='red',
              font='helvetica 8 bold ').place(x=20, y=200)
    elif(value==3):
        draw_iris_dataset.draw_x1_x4()
        Label(Master, text="X1 and X2 are descriminative between C1 and C2 && C3           ", bg='red',
              font='helvetica 8 bold ').place(x=20, y=180)
        Label(Master,
              text="and are barely descriminative between C3 and C1 && C2        ",
              bg='red',
              font='helvetica 8 bold ').place(x=20, y=200)
    elif(value==4):
        draw_iris_dataset.draw_x2_x3()
        Label(Master, text="X1 and X2 are descriminative between C1 and C2 && C3          ", bg='red',
              font='helvetica 8 bold ').place(x=20, y=180)
        Label(Master,
              text="but we can seperate between C2 and C3 hardly                  ",
              bg='red',
              font='helvetica 8 bold ').place(x=20, y=200)
    elif(value==5):
        draw_iris_dataset.draw_x2_x4()
        Label(Master, text="X1 and X2 are descriminative between C1 and C2 && C3          ", bg='red',
              font='helvetica 8 bold ').place(x=20, y=180)
        Label(Master,
              text="and are barely descriminative between C3 and C1 && C2        ",
              bg='red',
              font='helvetica 8 bold ').place(x=20, y=200)
    else:
        draw_iris_dataset.draw_x3_x4()
        Label(Master, text="X1 and X2 are descriminative between C1 and C2 && C3         ", bg='red',
              font='helvetica 8 bold ').place(x=20, y=180)
        Label(Master,
              text="and are barely descriminative between C3 and C1 && C2        ",
              bg='red',
              font='helvetica 8 bold ').place(x=20, y=200)
Button(Master,text="Draw My Data",font='helvetica 8 bold ',width=20,command=lambda :draw_button(select.get())).place(x=250,y=30)




#---------------------------------------------------------------------------









##################################  User Interactions ####################################

F_menuclicked=StringVar()
OptionMenu(Master,F_menuclicked,"X1 and X2","X1 and X3","X1 and X4","X2 and X3","X2 and X4","X3 and X4").place(x=600,y=20)
L1=Label(Master,text="Features")
L1.place(x=780,y=25)
L1.configure(font='helvetica 12 bold ')

C_menuclicked=StringVar()
OptionMenu(Master,C_menuclicked,"Iris-setosa and Iris-versicolor","Iris-setosa and Iris-virginica","Iris-versicolor and Iris-virginica").place(x=600,y=50)
L2=Label(Master,text="Classes")
L2.place(x=780,y=55)
L2.configure(font='helvetica 12 bold ')


Rate=Entry(Master)
Rate.place(x=600,y=85)
Rate.focus_set()
L3=Label(Master,text="Rate")
L3.place(x=780,y=85)
L3.configure(font='helvetica 12 bold ')


Epoch=Entry(Master)
Epoch.place(x=600,y=115)
Epoch.focus_set()
L4=Label(Master,text="Epoch")
L4.place(x=780,y=115)
L4.configure(font='helvetica 12 bold ')

bais_checked=IntVar()
Checkbutton(Master,text="Bais",font='helvetica 12 bold ',variable=bais_checked).place(x=600, y=145)

#--------------------------------------------------------------------------


############################  Algorithm Buttons  ######################################




W = None
b = None
loss = None
X_train = None
X_test = None
y_train = None
y_test = None
y2_pred = None
CM = None
acc = None


def Train_test_button(features, classes, Rate, epochs, bais):
    global W, b, loss, X_train, X_test, y_train, y_test, y2_pred
    W, b, loss, X_train, X_test, y_train,y_test,y2_pred=main.train_Test_NN(features, classes, Rate, epochs, bais)
Button(Master,text="Train",font='helvetica 8 bold ',bg='blue',height=3,width=30,command=lambda  :Train_test_button(F_menuclicked.get(),C_menuclicked.get(),Rate.get(),Epoch.get(),bais_checked.get())).place(x=670,y=240)


def Evaluate_button():
    CM, acc = main.evaluate_NN(y2_pred, y_test)
    acc = acc * 100
    Label(Master, text="My Matrix = ", font='helvetica 12 bold ').place(x=330, y=100)
    Label(Master, text="My Accurcay =   %", font='helvetica 12 bold ').place(x=330, y=150)
    Label(Master, text=CM, font='helvetica 12 bold ', bg='yellow').place(x=470, y=100)
    Label(Master, text=acc, font='helvetica 12 bold ', bg='yellow').place(x=470, y=150)
Button(Master, text="Evaluate", font='helvetica 8 bold ', bg='blue', height=3, width=30,
       command=lambda: Evaluate_button()).place(x=230, y=240)

def draw_line_button():
    main.draw_line(W, b, X_test)
Button(Master, text="Draw Line", font='helvetica 8 bold ', bg='blue', command=lambda: draw_line_button(),
       height=3, width=30).place(x=450, y=240)


#---------------------------------------------------------------------------------------



Master.mainloop()



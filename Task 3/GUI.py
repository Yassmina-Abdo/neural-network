from tkinter import *
import Algorithms
########################################## GUI INtERFACE ##############################################################################
Master=Tk()
Master.geometry("900x300")
#------------------------------------ Ebocs TextBox ----------------------------------------------------------------------
Epoch=Entry(Master)
Epoch.place(x=200,y=85)
L4=Label(Master,text="Epoch")
L4.place(x=100,y=85)
L4.configure(font='helvetica 12 bold ')
#------------------------------------ Ebocs TextBox ----------------------------------------------------------------------
#------------------------------------ L Rate TextBox ----------------------------------------------------------------------
Rate=Entry(Master)
Rate.place(x=200,y=115)
Rate.focus_set()
L3=Label(Master,text="Rate")
L3.place(x=100,y=115)
L3.configure(font='helvetica 12 bold ')
#------------------------------------ L Rate TextBox ----------------------------------------------------------------------
#------------------------------------ Hidden layer TextBox ----------------------------------------------------------------------
Hidden_layer=Entry(Master)
Hidden_layer.place(x=600,y=85)
L5=Label(Master,text="Hidden layer")
L5.place(x=500,y=85)
L5.configure(font='helvetica 12 bold ')
#------------------------------------ Hidden layer TextBox ----------------------------------------------------------------------
#------------------------------------ Neurons TextBox ----------------------------------------------------------------------
Neurons=Entry(Master)
Neurons.place(x=600,y=115)
L5=Label(Master,text="Neurons")
L5.place(x=500,y=115)
L5.configure(font='helvetica 12 bold ')
#------------------------------------ Neurons TextBox ----------------------------------------------------------------------
#------------------------------------  Activation Function Menue ---------------------------------------------------------------------
Activation_Function=StringVar()
OptionMenu(Master,Activation_Function,"Sigmoid","Hyperbolic").place(x=260,y=145)
L1=Label(Master,text="Activation Function")
L1.place(x=100,y=145)
L1.configure(font='helvetica 12 bold ')
#------------------------------------  Activation Function Menue ---------------------------------------------------------------------
#------------------------------------ Bias Check Box ----------------------------------------------------------------------
bais_checked=IntVar()
Checkbutton(Master,text="Bais",font='helvetica 12 bold ',variable=bais_checked).place(x=500, y=145)
#------------------------------------ Bias Check Box ----------------------------------------------------------------------
#------------------------------------ EVALUATE BUTTON -----------------------------------------------------------------------
def Evaluate_button():
    CM, acc = Algorithms.evaluate_NN(int(Epoch.get()),float(Rate.get()),int(Hidden_layer.get()),Neurons.get(),Activation_Function.get(),bais_checked.get())
    acc = acc * 100
    Label(Master, text="My Matrix = ", font='helvetica 12 bold ').place(x=600, y=200)
    Label(Master, text="My Accurcay = %", font='helvetica 12 bold ').place(x=600, y=270)
    Label(Master, text=CM, font='helvetica 12 bold ', bg='yellow').place(x=740, y=200)
    Label(Master, text=acc, font='helvetica 12 bold ', bg='yellow').place(x=740, y=270)
Button(Master, text="Evaluate", font='helvetica 8 bold ', bg='blue', height=3, width=30,command=lambda: Evaluate_button()).place(x=330, y=220)
#------------------------------------ EVALUATE BUTTON -----------------------------------------------------------------------

##########################################  GUI INtERFACE #######################################################################################
Master.mainloop()
import pytesseract
import cv2
from tkinter import *

root = Tk()
text = Text(root)
text.insert(INSERT, "Hello, Below is the text found on the image\n\n")




img = cv2.imread(r'ab.jpg')

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
tex = pytesseract.image_to_string(img)

text.insert(INSERT,tex)

print(text)
text.pack()
root.mainloop()
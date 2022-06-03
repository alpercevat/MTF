import sys
import matplotlib as plt
from tkinter import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk
import tkinter.filedialog
from tkinter.filedialog import askopenfilename # Open dialog box
from PIL import Image
import tkinter as tk
import cv2, time
from PIL import ImageTk
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np



def open_File():   

    filename = askopenfilename(filetypes=[("images","*.png")])
    img = cv2.imread(filename)
    #img = Image.open(filename)
    #img = ImageTk.PhotoImage(img)

    #cv2.imshow("Shapes", img) # I used cv2 to show image 
    #cv2.waitKey(0)

    roi = cv2.selectROI(img)
    roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    #plt.title("Line graph")
    #plt.imshow(roi_cropped)
    #plt.show()

    #cv2.imwrite("crop.jpeg",roi_cropped)
    #hold window
    #cv2.waitKey()
    #filename.close()

    ########################### ESF ######################################    
    #print(roi_cropped)
    ımage_arr = np.array(roi_cropped)
    R, G, B = ımage_arr[:,:,0], ımage_arr[:,:,1], ımage_arr[:,:,2]
    ımgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    #print(ımgGray)
    #plt.title("Gray Scale")
    #plt.imshow(ımgGray)
    #plt.show()

    X = ımgGray.mean(0)
    #print("X: ",X)
    #print(X.shape[0])

    mu = np.sum(X)/X.shape[0]
    tmp = (X[:] - mu)**2
    #print("tmp: ",tmp)
    sigma = np.sqrt(np.sum(tmp)/X.shape[0])
    edge_function = (X[:] - mu)/sigma
            
    #edge_function = edge_function[::3]
    xesf = range(0,edge_function.shape[0])

    #print("Edge Function: ",edge_function)
    #print("x: ",x)

    #plt.figure()
    #plt.title(r'ESF')
    #plt.plot(xesf,edge_function,'-ob')
    #plt.show()
    #################################################################


    ########################### LSF ######################################
    lsf =  edge_function[:-2] - edge_function[2:]
    xlsf = range(0,lsf.shape[0])
    
    #plt.title("LSF")
    #plt.xlabel(r'PİXEL') ; plt.ylabel('Intensity')
    #plt.plot(xlsf,lsf,"-or")
    #plt.show()
    #################################################################

    ########################### MTF ######################################
    mtf = abs(np.fft.fft(lsf))
    mtf = mtf[:]/np.max(mtf)
    mtf = mtf[:len(mtf)//2]
    #mtf = np.array(sorted(mtf, reverse= True))
    #mtf = mtf[::2]

    ix = np.arange(mtf.shape[0]) / (mtf.shape[0])
    mtf_poly =  np.polyfit(ix, mtf, 6)
    poly = np.poly1d(mtf_poly)

    #plt.figure()
    #plt.title("MTF")
    #plt.xlabel(r'Frequency $[Cycles/Pixel]$') ; plt.ylabel('MTF')
    #p, = plt.plot(ix,mtf,'-or')
    #ll, = plt.plot(ix,poly(ix)) 
    #plt.legend([p,ll],["MTF values","Polynomial Fit"])
    #plt.grid()
    #plt.show()
    print(mtf)
    ###########################################################################
                        ## Modified MTF

    lsf_mod =  edge_function[:-1] - edge_function[1:]
    xlsf_mod = range(0,lsf_mod.shape[0])

    mtf_mod = abs(np.fft.fft(lsf_mod))
    mtf_mod = mtf_mod[:]/np.max(mtf_mod)
    mtf_mod = mtf_mod[:len(mtf_mod)//2]

    ix_mod = np.arange(mtf_mod.shape[0]) / (mtf_mod.shape[0])
    mtf_poly_mod =  np.polyfit(ix_mod, mtf_mod, 6)
    poly_mod = np.poly1d(mtf_poly_mod)

    print(mtf_mod)
   #################################################################
                                #PLOTS

    plt.title("Seçilen Bölge")
    plt.imshow(roi_cropped)
    plt.show()

    fig, ( (ax1, ax2), (ax3, ax4) ) = plt.subplots(2, 2 , figsize=(15, 9))


    ax1.plot(xesf,edge_function,'-.r')
    ax1.set_title("Edge Spread Function")
    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Pixel")
    ax1.grid()

    ax2.plot(xlsf,lsf,"-.b")
    ax2.set_title("Line Spread Function")
    ax2.set_ylabel("Intensity Change")
    ax2.set_xlabel("Pixel")
    ax2.grid()

    ax3.plot(ix,poly(ix))
    ax3.set_title("Modulation Transfer Function Polynom Fit")
    ax3.set_ylabel("MTF")
    ax3.set_xlabel("Frequency[Cycles/Pixel]")
    ax3.grid()

    ax4.plot(ix,mtf,'.-b')
    ax4.set_title("Modulation Transfer Function Data")
    ax4.set_ylabel("MTF")
    ax4.set_xlabel("Frequency[Cycles/Pixel]")
    ax4.grid()

    plt.show()

    ########################## PLOTS MODİFİED ################################

    #plt.title("Seçilen Bölge")
    #plt.imshow(roi_cropped)
    #plt.show()

    fig, ( (ax1, ax2), (ax3, ax4) ) = plt.subplots(2, 2 , figsize=(15, 9))


    ax1.plot(xesf,edge_function,'-.r')
    ax1.set_title("Edge Spread Function")
    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Pixel")
    ax1.grid()

    ax2.plot(xlsf_mod,lsf_mod,"-.b")
    ax2.set_title("Line Spread Function")
    ax2.set_ylabel("Intensity Change")
    ax2.set_xlabel("Pixel")
    ax2.grid()

    ax3.plot(ix_mod,poly_mod(ix_mod))
    ax3.set_title("Modulation Transfer Function Polynom Fit    (MODİFİED)")
    ax3.set_ylabel("MTF")
    ax3.set_xlabel("Frequency[Cycles/Pixel]")
    ax3.grid()

    ax4.plot(ix_mod,mtf_mod,'.-b')
    ax4.set_title("Modulation Transfer Function Data        (MODİFİED)")
    ax4.set_ylabel("MTF")
    ax4.set_xlabel("Frequency[Cycles/Pixel]")
    ax4.grid()

    plt.show()
    #################################################################

root = Tk()

frm = Frame(root)
frm.pack(side = BOTTOM, padx=15, pady = 15)

lbl = Label(root, text="Edge Spread Function", fg = "black", font = "Times")
lbl.pack()

lbl = Label(root, text="Line Spread Function", fg = "black", font = "Times")
lbl.pack()

lbl = Label(root, text="Modulation Transfer Function", fg = "black", font = "Times")
lbl.pack()


logo = tk.PhotoImage(file="C:/Users/Alper/Desktop/Logo11.png")

lbl = Label(root, 
             compound = tk.CENTER,
             image=logo).pack(side="bottom")

btn = Button(frm, text ="Resim Seç", command = open_File)
btn.pack(side = tk.LEFT)

btn2 = Button(frm, text ="Çıkış", command = lambda: exit() )
btn2.pack(side = tk.LEFT, padx =10)

root.title("MTF Hesaplama Uygulaması")
root.geometry("500x500")
root.mainloop()
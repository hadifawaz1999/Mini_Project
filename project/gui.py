from tkinter import *
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from tslearn.metrics import dtw


class GUI:

    def __init__(self):

        self.gui = Tk()
        self.gui.title("Digit Recognition")
        self.gui.geometry("1280x720")
        self.gui['bg'] = "white"

        # close button

        self.close_button = Button(
            self.gui, text="Close", command=self.close_button_command)
        self.close_button['relief'] = "raised"
        self.close_button['activeforeground'] = "red"
        self.close_button['activebackground'] = "white"
        self.close_button['fg'] = "black"
        self.close_button['padx'] = 10
        self.close_button['pady'] = 10

        self.close_button.place(x=1200, y=660)

        # load data button

        self.load_button = Button(
            self.gui, text="Load Data Mnist", command=self.load_button_command)
        self.load_button['relief'] = "raised"
        self.load_button['activeforeground'] = "green"
        self.load_button['activebackground'] = "white"
        self.load_button['fg'] = "black"
        self.load_button['padx'] = 10
        self.load_button['pady'] = 10

        self.load_button.place(x=30, y=30)

        # show random digit

        self.show_random_button = Button(
            self.gui, text="Show Random Digit", command=self.show_random_button_command)
        self.show_random_button['relief'] = "raised"
        self.show_random_button['activeforeground'] = "green"
        self.show_random_button['activebackground'] = "white"
        self.show_random_button['fg'] = "black"
        self.show_random_button['padx'] = 10
        self.show_random_button['pady'] = 10

        # show random 10 digits

        self.show_all_button = Button(
            self.gui, text="Show The 10 Digits", command=self.show_all_button_command)
        self.show_all_button['relief'] = "raised"
        self.show_all_button['activeforeground'] = "green"
        self.show_all_button['activebackground'] = "white"
        self.show_all_button['fg'] = "black"
        self.show_all_button['padx'] = 10
        self.show_all_button['pady'] = 10

        # load test digits

        self.load_test_digit_button = Button(
            self.gui, text="Load Test Digit", command=self.load_test_digit_button_command)
        self.load_test_digit_button['relief'] = "raised"
        self.load_test_digit_button['activeforeground'] = "green"
        self.load_test_digit_button['activebackground'] = "white"
        self.load_test_digit_button['fg'] = "black"
        self.load_test_digit_button['padx'] = 10
        self.load_test_digit_button['pady'] = 10

        # dtw button

        self.predict_random_digit_by_dtw_button = Button(
            self.gui, text="Predict Random Digit By DTW", command=self.predict_random_digit_by_dtw_button_command)
        self.predict_random_digit_by_dtw_button['relief'] = "raised"
        self.predict_random_digit_by_dtw_button['activeforeground'] = "green"
        self.predict_random_digit_by_dtw_button['activebackground'] = "white"
        self.predict_random_digit_by_dtw_button['fg'] = "black"
        self.predict_random_digit_by_dtw_button['padx'] = 10
        self.predict_random_digit_by_dtw_button['pady'] = 10

        # dtw test image button

        self.predict_by_dtw_button = Button(
            self.gui, text="Predict Test Image By DTW", command=self.predict_by_dtw_button_command)
        self.predict_by_dtw_button['relief'] = "raised"
        self.predict_by_dtw_button['activeforeground'] = "green"
        self.predict_by_dtw_button['activebackground'] = "white"
        self.predict_by_dtw_button['fg'] = "black"
        self.predict_by_dtw_button['padx'] = 10
        self.predict_by_dtw_button['pady'] = 10

        # ED button

        self.predict_random_digit_by_ed_button = Button(
            self.gui, text="Predict Random Digit By ED", command=self.predict_random_digit_by_ed_button_command)
        self.predict_random_digit_by_ed_button['relief'] = "raised"
        self.predict_random_digit_by_ed_button['activeforeground'] = "green"
        self.predict_random_digit_by_ed_button['activebackground'] = "white"
        self.predict_random_digit_by_ed_button['fg'] = "black"
        self.predict_random_digit_by_ed_button['padx'] = 10
        self.predict_random_digit_by_ed_button['pady'] = 10

        # ED test image button

        self.predict_by_ed_button = Button(
            self.gui, text="Predict Test Image By ED", command=self.predict_by_ed_button_command)
        self.predict_by_ed_button['relief'] = "raised"
        self.predict_by_ed_button['activeforeground'] = "green"
        self.predict_by_ed_button['activebackground'] = "white"
        self.predict_by_ed_button['fg'] = "black"
        self.predict_by_ed_button['padx'] = 10
        self.predict_by_ed_button['pady'] = 10

    def start_loop(self):
        self.gui.mainloop()

    def close_button_command(self):
        self.gui.quit()

    def load_button_command(self):
        self.data_path = "/media/hadi/laban/data_sets/keras/mnist/"
        
        self.xtrain = np.load(self.data_path+"x_train.npy")
        self.xtrain = np.asarray(self.xtrain, dtype=np.float64)

        self.xtrain = self.xtrain/255

        self.ytrain = np.load(self.data_path+"y_train.npy")
        self.ytrain = np.asarray(self.ytrain, dtype=np.float64)

        self.xtest = np.load(self.data_path+"x_test.npy")
        self.xtest = np.asarray(self.xtest, dtype=np.float64)

        self.xtest = self.xtest/255

        self.ytest = np.load(self.data_path+"y_test.npy")
        self.ytest = np.asarray(self.ytest, dtype=np.float64)

        print("xtrain: ", self.xtrain.shape, "ytrain: ", self.ytrain.shape,
              "xtest: ", self.xtest.shape, "ytest: ", self.ytest.shape)

        self.show_random_button.place(x=200, y=30)
        self.show_all_button.place(x=370, y=30)
        self.load_test_digit_button.place(x=540, y=30)
        self.predict_random_digit_by_dtw_button.place(x=30, y=110)
        self.predict_random_digit_by_ed_button.place(x=30,y=200)

    def show_random_button_command(self):
        index = random.randint(a=0, b=int(self.xtrain.shape[0]))
        plt.imshow(self.xtrain[index], cmap="gray")
        plt.show()

    def show_all_button_command(self):
        fig, sub = plt.subplots(nrows=4, ncols=3, figsize=(10, 8))
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        index = 0
        subx = 1
        suby = 0
        sub[0, 0].axis('off')
        sub[0, 2].axis('off')
        for i in range(int(self.xtrain.shape[0])):
            if self.ytrain[i] == digits[index]:
                if index == 0:
                    sub[0, 1].imshow(self.xtrain[i], cmap="gray")
                else:
                    sub[subx, suby].imshow(self.xtrain[i], cmap="gray")
                    if suby == 2:
                        suby = 0
                        subx += 1
                    else:
                        suby += 1
                index += 1
                if index == 10:
                    break
        plt.show()

    def load_test_digit_button_command(self):
        try:

            self.predict_by_dtw_button.place(x=280, y=110)
            self.predict_by_ed_button.place(x=280,y=200)
            img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
            self.test_digit = np.asarray(img, dtype=np.float64)
            print("test image: ", self.test_digit.shape)
            plt.imshow(self.test_digit, cmap="gray")
            plt.show()
        except:
            tkinter.messagebox.showerror(
                "Error !!!!", "Create test image first")

    def predict_by_dtw_button_command(self):
        ytrain_copy = self.ytrain.copy()
        xtrain_copy = self.xtrain.copy()
        ytrain_copy.shape = (-1, 1)
        xtrain_copy.shape = (60000, -1)
        data_combined = np.concatenate((ytrain_copy, xtrain_copy), axis=1)

        np.random.shuffle(data_combined)
        img = self.test_digit.copy()
        img.shape = (-1,)
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        index = 0
        chosen = np.zeros(shape=(10, 28*28))
        chosen_label = np.zeros(shape=(10,))
        for i in range(int(data_combined.shape[0])):
            if data_combined[i, 0] == digits[index]:
                chosen[index] = data_combined[i, 1:]
                chosen_label[index] = data_combined[i, 0]
                index += 1
                if index == 10:
                    break
                np.random.shuffle(data_combined)
        dtw_distances = []
        for i in range(10):
            dtw_distances.append(dtw(img, chosen[i]))
        tkinter.messagebox.showinfo(
            "Prediction By DTW:", "The test image is predicted as "+str(np.argmin(dtw_distances)))

    def predict_random_digit_by_dtw_button_command(self):
        ytrain_copy = self.ytrain.copy()
        xtrain_copy = self.xtrain.copy()
        ytrain_copy.shape = (-1, 1)
        xtrain_copy.shape = (60000, -1)
        data_combined = np.concatenate((ytrain_copy, xtrain_copy), axis=1)

        np.random.shuffle(data_combined)
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        index = 0
        chosen = np.zeros(shape=(10, 28*28))
        chosen_label = np.zeros(shape=(10,))
        for i in range(int(data_combined.shape[0])):
            if data_combined[i, 0] == digits[index]:
                chosen[index] = data_combined[i, 1:]
                chosen_label[index] = data_combined[i, 0]
                index += 1
                if index == 10:
                    break
                np.random.shuffle(data_combined)
        dtw_distances = []
        index = random.randint(a=0, b=int(data_combined.shape[0]))
        img = data_combined[index, 1:]
        label = data_combined[index, 0]
        for i in range(10):
            dtw_distances.append(dtw(img, chosen[i]))
        tkinter.messagebox.showinfo("Prediction By DTW:", "The random digit selected ("+str(int(
            label))+") is predicted as "+str(np.argmin(dtw_distances)))

    def predict_random_digit_by_ed_button_command(self):
        ytrain_copy = self.ytrain.copy()
        xtrain_copy = self.xtrain.copy()
        ytrain_copy.shape = (-1, 1)
        xtrain_copy.shape = (60000, -1)
        data_combined = np.concatenate((ytrain_copy, xtrain_copy), axis=1)

        np.random.shuffle(data_combined)
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        index = 0
        chosen = np.zeros(shape=(10, 28*28))
        chosen_label = np.zeros(shape=(10,))
        for i in range(int(data_combined.shape[0])):
            if data_combined[i, 0] == digits[index]:
                chosen[index] = data_combined[i, 1:]
                chosen_label[index] = data_combined[i, 0]
                index += 1
                if index == 10:
                    break
                np.random.shuffle(data_combined)
        ed_distances = []
        index = random.randint(a=0,b=int(data_combined.shape[0]))
        img = data_combined[index,1:]
        label = int(data_combined[index,0])
        for i in range(10):
            ed_distances.append(np.sqrt(np.sum(np.square(img-chosen[i]))))
        tkinter.messagebox.showinfo("Prediction By ED:", "The random digit selected ("+str(int(
            label))+") is predicted as "+str(np.argmin(ed_distances)))

    def predict_by_ed_button_command(self):
        ytrain_copy = self.ytrain.copy()
        xtrain_copy = self.xtrain.copy()
        ytrain_copy.shape = (-1, 1)
        xtrain_copy.shape = (60000, -1)
        data_combined = np.concatenate((ytrain_copy, xtrain_copy), axis=1)

        np.random.shuffle(data_combined)
        img = self.test_digit.copy()
        img.shape = (-1,)
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        index = 0
        chosen = np.zeros(shape=(10, 28*28))
        chosen_label = np.zeros(shape=(10,))
        for i in range(int(data_combined.shape[0])):
            if data_combined[i, 0] == digits[index]:
                chosen[index] = data_combined[i, 1:]
                chosen_label[index] = data_combined[i, 0]
                index += 1
                if index == 10:
                    break
                np.random.shuffle(data_combined)
        ed_distances = []
        for i in range(10):
            ed_distances.append(np.sqrt(np.sum(np.square(img-chosen[i]))))
        tkinter.messagebox.showinfo(
            "Prediction By ED:", "The test image is predicted as "+str(np.argmin(ed_distances)))

if __name__ == "__main__":
    my_gui = GUI()
    my_gui.start_loop()
    exit()

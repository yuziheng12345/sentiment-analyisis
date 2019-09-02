# coding:utf-8
import tkinter
from tkinter import messagebox
from keras.models import load_model
import Data_process.testData_process as tp
from metrics import fmeasure,recall,precision

win = tkinter.Tk()
win.title("Sentiment analysis for WeChat articles")
win.geometry("400x400+200+50")
canvas = tkinter.Canvas(win,
                        width=1000,  # 指定Canvas组件的宽度
                        height=1000,  # 指定Canvas组件的高度
                        bg='LightGreen')  # 指定Canvas组件的背景色
canvas.pack()

'''点击事件处理'''
def func():
    content = input_text.get("0.0", "end")
    model = load_model("./model/weights.best.hdf5",custom_objects={'fmeasure': fmeasure,'recall':recall,'precision':precision})
    vec_content = tp.read_testdata_gui(content)
    res = model.predict(vec_content).tolist()[0]
    neg_score = res[0]
    pos_score = res[1]
    print(neg_score)
    print(pos_score)
    sen=''
    if pos_score>=neg_score:
        sen = '正面'
    else:
        sen = '负面'
    #分析结果显示
    messagebox.showinfo('情感分析结果', '情感倾向:'+sen+'\n'+'正面概率：'+str(pos_score)[:5]+'\n'+'负面概率：'+str(neg_score)[:5])

input_label = tkinter.Label(text='请输入微信文章:', bg='LightGreen',font=("Arial", 13))
input_label.place(relx=0.2, rely=0.05, relwidth=0.6, relheight=0.1)

input_text = tkinter.Text(win,width=40,height=10)
input_text.place(relx=0.1, rely=0.15, relwidth=0.8, relheight=0.6)


submit_button = tkinter.Button(win, text="提交", command=func)
submit_button.place(relx=0.4, rely=0.8, relwidth=0.2, relheight=0.1)

win.mainloop()
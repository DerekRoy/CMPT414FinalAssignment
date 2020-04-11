from kivy.app import App
from PIL import Image
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Fbo
from kivy.config import Config
import numpy as np
import cnn
import sys
import os
Config.set('graphics','resizable',0)

from kivy.core.window import Window

Window.size = (100, 100)

image = np.zeros((28,28))
c = cnn.CNN(image)

file = ""
if getattr(sys, 'frozen', False):
    file=os.path.join(sys._MEIPASS, "8x8filters50kimages")
else:
    file="8x8filters50kimages"

c.load_model(file)



class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=5)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def on_touch_up(self, touch):
        self.export_to_png(filename='temp.png')
        pil_img = Image.open('temp.png').convert('L')
        pil_img.thumbnail((28, 28), Image.ANTIALIAS)
        img = np.asarray(pil_img)
        result = c.feed_forward(img)
        print('Recognized:', np.where(result == 1.)[0][0])
        self.canvas.clear()


class MyPaintApp(App):

    def build(self):
        return MyPaintWidget()


if __name__ == '__main__':
    MyPaintApp().run()

from tkinter import *
from tkinter import ttk
import model
import image_extraction
from PIL import Image, ImageDraw , ImageTk

class App():

    def __init__(self):
        self.root = Tk()
        self.root.title("MNIST Convolutional Neural Network")
        self.layer = ''
        self.style = ttk.Style()
        self.style.configure('TButton',
                             padding=(10, 5, 10, 5),
                             font=('Arial', 12),
                             background='white',  
                             foreground='black',  
                             borderwidth=2,
                             relief="flat")

        self.button_frame = ttk.Frame(self.root, padding=(10, 5, 10, 5), style='TButton')
        self.button_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.c = Canvas(self.root, bg='black', width=224, height=224)
        self.c.grid(row=1, column=0)

        self.erase_button = ttk.Button(self.button_frame, text='erase', command=self.erase)
        self.erase_button.grid(row=0, column=0)

        self.w = Canvas(self.root, bg='grey', width=800, height=600)
        self.w.grid(row=1, column=1, rowspan=5)

        self.predict_button = ttk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=0, column=1)

        self.view_filters_button = ttk.Button(self.button_frame, text='View Filters', command=self.view_filters)
        self.view_filters_button.grid(row=0, column=2)

        self.view_activations_button = ttk.Button(self.button_frame, text='View Feature Maps', command=self.view_activations)
        self.view_activations_button.grid(row=0, column=3)

        self.result_label = Label(self.root, text="Prediction Result: ")
        self.result_label.grid(row=2, column=0)
        self.var = IntVar()
        self.radio_button2 = Radiobutton(self.root, text="1-layer convolution", variable=self.var, value= 1)
        self.radio_button2.grid(row=4,column=0)
        self.radio_button3 = Radiobutton(self.root, text="2-layer convolution", variable=self.var, value= 2)
        self.radio_button3.grid(row=5,column=0)
        self.setup()
        self.root.mainloop()


    def predict(self):
        model_type = self.var.get()
        self.save_as_png("pred")
        pred = model.run(model_type)
        result_text = f"Prediction Result: {pred[0]} with {pred[1]:,.3f}%"
        self.result_label.config(text=result_text)
        if model_type == 1:
            self.layer = 'conv2d'
        else:
            self.layer = 'conv2d_1'
            model.save_activations(model.model, 'conv2d_2','pred.png')
            model.save_filters(model.model, 'conv2d_2')
        model.save_filters(model.model, self.layer)
        model.save_activations(model.model, self.layer,'pred.png')

        image_extraction.get_combined_image(model_type)
        original_image = Image.open("combined_image.png").resize((800,600), Image.LANCZOS)
        image = ImageTk.PhotoImage(original_image)
        self.w.image = image
        self.w.create_image(0, 0, anchor=NW, image=image)

    def view_filters(self):
        model.visualize_filters(model.model, self.layer)

    def view_activations(self):
        model.visualize_activations(model.model, self.layer, 'pred.png')
        
    def save_as_png(self, file_path):
        canvas_width = self.c.winfo_width()
        canvas_height = self.c.winfo_height()

        image = Image.new('RGB', (canvas_width, canvas_height), 'black')
        draw = ImageDraw.Draw(image)

        for item in self.c.find_all():
            x1, y1, x2, y2 = self.c.coords(item)
            item_options = self.c.itemconfig(item)
            line_color = item_options.get('fill', ('white',))[-1]
            draw.line([x1, y1, x2, y2], fill=line_color, width=self.line_width)

        image.save(file_path + '.png')

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 16
        self.color = 'white'
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def erase(self):
        self.c.delete("all")

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

if __name__ == "__main__":
    App()

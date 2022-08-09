from operator import index
from tkinter import Grid
from tkinter.tix import ROW
from utils import *
import tensorflow as tf
import numpy as np
# from workonpygame import HEIGHT, WHITE, WIDTH


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test , y_test) = mnist.load_data()
x_train.shape

x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)


img_size = 28
x_trainer = np.array(x_train).reshape(-1,img_size,img_size,1)
x_tester = np.array(x_test).reshape(-1,img_size,img_size,1)

from tensorflow import keras
model = keras.models.load_model('digit_recogniser_model.h5')


WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Rec")

def init_grid(rows, cols, color):
    grid = []
    tensorgridinit = []
    for i in range(rows):
        grid.append([])
        tensorgridinit.append([])
        for _ in range(cols):
            grid[i].append(color)
            tensorgridinit[i].append(255)

    return tensorgridinit, grid

def draw_grid(win, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            pygame.draw.rect(win, (pixel), (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    if DRAW_GRIS_LINES:
        for i in range(ROWS + 1):
            pygame.draw.line(win, BLACK, (0, i * PIXEL_SIZE), (WIDTH, i * PIXEL_SIZE))
        for i in range(COLS + 1):
            pygame.draw.line(win, BLACK, (i * PIXEL_SIZE, 0), (i * PIXEL_SIZE, HEIGHT - TOOLBAR_HEIGHT))




def draw(win, grid, buttons):
    win.fill(BG_COLOR)
    draw_grid(win, grid)
    for button in buttons:
        button.draw(win)
    pygame.display.update()

def get_row_col_pos(pos):
    y, x = pos
    row = x // PIXEL_SIZE
    col = y // PIXEL_SIZE

    if row >= ROWS:
        raise IndexError

    return row, col



run = True
clock = pygame.time.Clock()
tensorgrid, grid = init_grid(ROWS, COLS, BG_COLOR)

drawing_color = BLACK


button_y = HEIGHT - TOOLBAR_HEIGHT/2 - 25

buttons = [
    Button(10, button_y, 50, 50, WHITE, 'Clear', BLACK),
    # Button(80, button_y, 50, 50, WHITE, 'Send', BLACK),
]

def makeprediction(resize,win):
    
    new_img = tf.keras.utils.normalize(resize, axis=1)
    new_img = np.array(new_img).reshape(-1,img_size,img_size,1)
    predictions = model.predict(new_img)
    print(str(np.argmax(predictions)))


while run:
    clock.tick(FPS)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            try:
                row, col = get_row_col_pos(pos)
                grid[row][col] = drawing_color
                tensorgrid[row][col] = 0
                # resize = ([255, 255, 255, 254, 255, 254, 255, 254, 254, 254, 255, 255, 255, 253, 255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [253, 255, 255, 255, 251, 255, 253, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 254, 255, 255, 253, 253, 255, 254, 254, 255, 254, 253, 255, 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [253, 254, 254, 255, 250, 255, 255, 255, 253, 255, 252, 255, 255, 254, 255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 254, 255, 255, 252, 255, 0, 0, 1, 0, 0, 0, 0, 2, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [248, 255, 255, 253, 255, 255, 0, 0, 254, 253, 254, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 252, 255, 252, 255, 0, 255, 253, 255, 255, 252, 253, 252, 255, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 254, 254, 253, 255, 255, 253, 255, 255, 251, 254, 255, 254, 255, 254, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 255, 252, 253, 255, 255, 0, 254, 253, 255, 255, 251, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 255, 255, 255, 252, 1, 2, 255, 250, 252, 255, 255, 252, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 254, 252, 255, 251, 255, 254, 255, 255, 1, 252, 255, 254, 255, 253, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 255, 255, 252, 254, 0, 255, 253, 255, 255, 255, 254, 254, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 254, 250, 255, 255, 250, 253, 255, 255, 0, 254, 254, 255, 255, 251, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 253, 251, 255, 255, 254, 0, 2, 253, 255, 253, 250, 255, 252, 253, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 255, 254, 255, 0, 253, 253, 255, 255, 254, 255, 252, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 0, 0, 255, 255, 253, 255, 255, 253, 254, 255, 255, 255, 255, 255], [255, 254, 254, 255, 255, 253, 255, 255, 255, 254, 255, 253, 253, 5, 0, 255, 254, 254, 255, 255, 255, 252, 255, 252, 255, 255, 255, 255], [254, 254, 255, 252, 254, 255, 251, 255, 255, 255, 249, 3, 3, 0, 252, 255, 255, 255, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255], [255, 255, 253, 255, 255, 253, 255, 1, 0, 0, 1, 0, 251, 255, 255, 254, 254, 253, 255, 254, 255, 255, 252, 251, 255, 255, 255, 255], [252, 254, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 252, 255, 255, 255, 254, 255, 255, 252, 255, 255, 255, 255, 255, 255, 255], [255, 254, 251, 255, 252, 255, 255, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 5, 250, 255, 254, 255, 255, 255, 255], [252, 255, 255, 255, 253, 255, 252, 255, 254, 255, 255, 255, 255, 252, 255, 254, 252, 255, 255, 255, 252, 255, 255, 255, 255, 255, 255, 255], [255, 255, 254, 252, 255, 252, 255, 255, 255, 255, 253, 252, 255, 255, 254, 255, 255, 254, 255, 252, 255, 255, 246, 255, 255, 255, 255, 255], [255, 255, 255, 255, 252, 254, 255, 255, 255, 251, 255, 255, 253, 255, 255, 255, 251, 255, 255, 254, 254, 255, 255, 254, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255] )
                makeprediction(tensorgrid, WIN)
            except IndexError:
                for button in buttons:
                    if not button.clicked(pos):
                        continue
                    

                    if button.text == 'Clear':
                        tensorgrid, grid = init_grid(ROWS, COLS, BG_COLOR)
                    # if button.text == 'Send':
                    #     makeprediction(tensorgrid)



    draw(WIN, grid, buttons)

pygame.quit()


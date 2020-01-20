import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import pygame
import os
import subprocess
import drowing, physic
from PIL import ImageTk, Image
import warnings
warnings.filterwarnings("ignore")

def start_demonstration():
    global first_load
    screen.fill(pygame.Color(240, 240, 240))
    work_frame.tkraise()
    amplitude_scale['state'] = 'normal'
    period_scale['state'] = 'normal'
    number_of_molecules_scale['state'] = 'normal'
    molecule_radius_scale['state'] = 'normal'
    molecule_mass_scale['state'] = 'normal'
    initial_temp_scale['state'] = 'normal'
    rel_scale['state'] = 'normal'
    first_load = True


def go_to_menu():
    global play_pause_button, Model_is_run
    Model_is_run = False
    menu_frame.tkraise()
    play_pause_button['text'] = 'Старт'


def end():
    global Done, Model_is_run
    Done = True
    Model_is_run = False


def theor():
    subprocess.Popen(['.\data\СФ.pdf'], shell=True)
    #os.startfile('data/СФ.pdf')

def alt_tab(*args):
    root.overrideredirect(False)
    pygame.time.delay(1000)

def authors():
    authors_frame.tkraise()


def left(*args):
    global pages, page
    page.set((page.get() - 2) % n_page + 1)
    pages[page.get() - 1].tkraise()


def right(*args):
    global pages, page
    page.set((page.get()) % n_page + 1)
    pages[page.get() - 1].tkraise()


def play_pause():
    global play_pause_button, menu_button, restart_button, \
        Model_is_run, var_targ_tg, var_ratio_tg
    if play_pause_button['text'] == 'Старт':
        play_pause_button['text'] = 'Продолжить'
        restart()
    elif play_pause_button['text'] == 'Продолжить':
        Model_is_run = True
        play_pause_button['text'] = 'Пауза'
        amplitude_scale['state'] = 'disabled'
        period_scale['state'] = 'disabled'
        number_of_molecules_scale['state'] = 'disabled'
        molecule_radius_scale['state'] = 'disabled'
        molecule_mass_scale['state'] = 'disabled'
        initial_temp_scale['state'] = 'disabled'
        rel_scale['state'] = 'disabled'
        cycle()
    else:
        Model_is_run = False
        play_pause_button['text'] = 'Продолжить'
        amplitude_scale['state'] = 'normal'
        period_scale['state'] = 'normal'
        number_of_molecules_scale['state'] = 'normal'
        molecule_radius_scale['state'] = 'normal'
        molecule_mass_scale['state'] = 'normal'
        initial_temp_scale['state'] = 'normal'
        rel_scale['state'] = 'normal'
        """
        if timeline.shape[0] >= 2:
            model.fit(timeline.reshape(-1, 1),
                      mean_speed)

            m = model.predict(np.array([[timeline[0]]]))[0]
            tg = (model.predict(np.array([[timeline[-1]]]))[0] - m) / \
                 (timeline[-1] - timeline[0])
            drowing.plot(screen, timeline, temp,
                         mark_font,
                         energy_plot_box, 20, 20, 'К')
            drowing.plot(screen, timeline, mean_speed,
                         mark_font,
                         speed_plot_box, 20, 20, 'м/с', tg, m)
            var_tg.set('{:.2f}'.format(tg * 1000))
            var_ratio_tg.set('{:.2f}'.format(tg * 1000 / targ_tg))
        """



def sin_move(x):
    global T, rel, A
    tmp = divmod(x, T)[1]
    if tmp <= T * rel:
        return A * (1 - np.cos(np.pi * tmp / T / rel)) / 2
    else:
        return A * (1 - np.cos(
            np.pi * (tmp - T * (2 * rel - 1)) / T / (1 - rel))) / 2


def sin_speed(x):
    global T, rel, A
    tmp = divmod(x, T)[1]
    if tmp <= T * rel:
        return A * np.sin(np.pi * tmp / T / rel) * np.pi / T / rel / 2
    else:
        return A * np.sin(
            np.pi * (tmp - T * (2 * rel - 1)) / T / (1 - rel)) \
               * np.pi / T / (1 - rel) / 2


def lin_move(x):
    global T, rel, A
    tmp = divmod(x, T)[1]
    if tmp <= T * rel:
        return A * tmp / T / rel
    else:
        return A * (T - tmp) / T / (1 - rel)


def lin_speed(x):
    global T, rel, A
    tmp = divmod(x, T)[1]
    if tmp <= T * rel:
        return A / T / rel
    else:
        return - A / T / (1 - rel)


def restart():
    global Model_is_run, var_R, R, R_lim, var_N, N, N_lim, var_A, A, \
        A_lim, var_T, T, T_lim, coord, speed, bord_box, piston_box, \
        piston_coord, piston_speed, t, mask, temp, timeline, \
        play_pause_button, mean_speed, var_mass, var_initial_temp, \
        mass, initial_temp, rel, plot_flag, max_speed, targ_tg,\
        var_targ_tg, mean_dt, varlambda, trust_lim, stabilized, \
        tg_hist, var_ratio_tg, var_tg
    stabilized = False
    Model_is_run = True
    plot_flag = False
    """
    limits = (R_lim, N_lim, A_lim, T_lim, mass_lim, initial_temp_lim)
    params = (var_R, var_N, var_A, var_T, var_mass, var_initial_temp)
    for i in range(len(params)):
        params[i].set(max(limits[i][0],
                          min(params[i].get(), limits[i][1])))
    """
    R = int(var_R.get() * 2 * max_amplitude)
    N = var_N.get()
    A = int(var_A.get() * 2 * max_amplitude)
    T = float(var_T.get())
    var_ratio_tg.set(0.0)
    var_tg.set(0.0)
    rel = var_rel.get() / (var_rel.get() + 1)
    mass = var_mass.get()
    initial_temp = var_initial_temp.get()
    varlambda = (line_box[3][1] - line_box[0][1]) * \
                (line_box[1][0] - line_box[0][0] - A / 2) / np.pi / \
                np.sqrt(2) / 2 / R / N
    varlambda = (line_box[1][0] - line_box[0][0] - A / 2) * np.pi
    trust_lim = varlambda * (1 - trust_index ** (1 / N))
    if type_of_movement.get() == 0:
        mps = (A * np.pi / T) ** 2 / 2
    else:
        mps = (A / T) ** 2 / rel / (1 - rel)
    targ_tg = mps / varlambda * size_coef * 1000 * 3 / 2
    var_targ_tg.set('{:.2f}'.format(targ_tg))
    piston_box[2] = 0
    bord_box[0][0] = indent
    coord = np.hstack((
        np.linspace(bord_box[0][0], bord_box[1][0], N).reshape(N, 1),
        np.random.randint(bord_box[0][1], bord_box[1][1],
                          size=(N, 1)))).astype(float)
    speed = np.random.normal(0, 5, (N, 2))
    if type_of_movement.get() == 0:
        piston_coord = sin_move
        piston_speed = sin_speed
    elif type_of_movement.get() == 1:
        piston_coord = lin_move
        piston_speed = lin_speed
    max_speed = A / T * min(rel, 1 - rel)
    t = 0
    mean_dt = 10
    tg_hist = np.empty(0)
    scale_temp = physic.temperature(speed, N, mass, size_coef)
    speed *= np.sqrt(initial_temp / scale_temp)
    temp = np.empty(shape=(0,))
    mean_speed = np.empty(shape=(0,))
    timeline = np.empty(shape=(0,))
    mask = np.zeros((coord.shape[0], coord.shape[0]), dtype=bool)
    play_pause_button['text'] = 'Продолжить'
    play_pause()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
def cycle():
    global Model_is_run, line_box, piston_box, coord, speed, t, mask, \
        bord_box, temp, timeline, mean_speed, plot_flag, tg, m,\
        start_time, var_ratio_tg, mean_dt, varlambda, trust_lim,\
        play_pause_button, stabilized, tg_hist
    clock.tick(fps)
    while Model_is_run:
        if root.focus_get() != None:
            root.overrideredirect(True)
        else:
            root.overrideredirect(False)
            root.iconify()
        dt_scale = float(var_dt_scale.get())
        dt = clock.tick(fps) / 1000 * dt_scale
        pygame.display.update()
        t += dt
        bord_box[0][0] = piston_coord(t) + indent
        coord, speed, mask = physic.iteration(coord, speed, dt, mask,
                                              bord_box, piston_speed(t),
                                              R)
        piston_box[2] = piston_coord(t)
        drowing.draw_sys(coord, screen, line_box, piston_box,
                         indent // 2, R,
                         clock.get_fps(), mark_font, timeline.shape[0])
        if not plot_flag:
            if np.mean(np.sqrt(np.sum(speed ** 2, axis=1))) >= \
                    5 * max_speed:
                start_time = t
                plot_flag = True
        else:
            temp = np.hstack((temp, physic.temperature(speed, N, mass,
                                                       size_coef)))
            mean_speed = np.hstack((mean_speed, size_coef *
                                    np.mean(np.sqrt(np.sum(speed ** 2,
                                                           axis=1)))))
            timeline = np.hstack((timeline, t - start_time))
        """
        if timeline.shape[0] == 2000:
            #temp = (temp[::2] + temp[1::2]) / 2
            temp = temp[1::]
            #mean_speed = (mean_speed[::2] + mean_speed[1::2]) / 2
            mean_speed = mean_speed[1::]
            #timeline = (timeline[::2] + timeline[1::2]) / 2
            timeline = timeline[1::]
            start_time += timeline[0]
            timeline = timeline - timeline[0]
        """
        if timeline.shape[0] == 2:
            tg = (mean_speed[1] - mean_speed[0]) / \
                 (timeline[1] - timeline[0])
            m = mean_speed[0] - tg * timeline[0]
        if timeline.shape[0] > 2:
            """
            model.fit(timeline.reshape(-1, 1),
                      mean_speed)

            m = model.predict(np.array([[timeline[0]]]))[0]
            tg = (model.predict(np.array([[timeline[-1]]]))[0] - m) / \
                 (timeline[-1] - timeline[0])
            #tg += (new_tg - tg) * learning_rate
            #m += (new_m - m) * learning_rate
            var_tg.set('{:.2f}'.format(tg * 100))
            var_ratio_tg.set('{:.2f}'.format(tg * 100 / targ_tg))
            """
            #var_ratio_tg.set('{:.6f}'.format(dt))
            if timeline.shape[0] > 3000:
                err = np.mean(np.abs(tg_hist * 1000 / targ_tg - 1))
                tg_hist = tg_hist[1:]
                if not stabilized and err < 0.1:
                    stabilized = True
                if stabilized and err > 0.15:
                    play_pause()
                    play_pause_button['text'] = 'Старт'

            if timeline.shape[0] > 2000:
                add_val = 2000 - timeline.shape[0] % 2000
                step = timeline.shape[0] // 2000
                work_mean_speed = np.mean(np.pad(mean_speed,
                                                 (0, add_val),
                                                 'constant').
                                          reshape(2000, -1), axis=1)
                work_temp = np.mean(np.pad(temp, (0, add_val),
                                           'constant').
                                          reshape(2000, -1), axis=1)
                work_time = timeline[::step][:2000]
                lim_shape = int(np.ceil(timeline.shape[0] /
                                        np.ceil(timeline.shape[0] /
                                                2000)))
                work_mean_speed = work_mean_speed[:lim_shape]
                work_temp = work_temp[:lim_shape]
                work_time = work_time[:lim_shape]
                """
                model.fit((timeline[-2000:] - timeline[-2000]).reshape(-1, 1),
                          mean_speed[-2000:])

                new_m = model.predict(np.array([[0]]))[0]
                new_tg = (model.predict(np.array([[timeline[-1] - timeline[-2000]]]))[
                          0] - new_m) / \
                     (timeline[-1] - timeline[-2000])
                var_tg.set('{:.2f}'.format(new_tg * 100))
                var_ratio_tg.set('{:.2f}'.format(new_tg * 100 / targ_tg))
                drowing.plot(screen, timeline[-2000:] - timeline[-2000], temp[-2000:], mark_font,
                             energy_plot_box, 20, 20, 'К')
                drowing.plot(screen, timeline[-2000:] - timeline[-2000], mean_speed[-2000:], mark_font,
                             speed_plot_box, 20, 20, 'м/с', new_tg, new_m)
                """
                model.fit(work_time.reshape(-1, 1), work_mean_speed)
                new_m = model.predict(np.array([[0]]))[0]
                new_tg = (model.predict(
                    np.array([[work_time[-1]]]))[0] - new_m) / \
                         (work_time[-1] - work_time[0])
                var_tg.set('{:.2f}'.format(new_tg * 1000))
                var_ratio_tg.set(
                    '{:.2f}'.format(new_tg * 1000 / targ_tg))
                drowing.plot(screen, work_time,
                             work_temp, mark_font,
                             energy_plot_box, 20, 20, 'К')
                drowing.plot(screen, work_time,
                             work_mean_speed, mark_font,
                             speed_plot_box, 20, 20, 'м/с', new_tg,
                             new_m)
            else:
                model.fit(timeline.reshape(-1, 1),
                          mean_speed)

                new_m = model.predict(np.array([[timeline[0]]]))[0]
                new_tg = (model.predict(np.array([[timeline[-1]]]))[
                          0] - new_m) / \
                     (timeline[-1] - timeline[0])
                var_tg.set('{:.2f}'.format(new_tg * 1000))
                var_ratio_tg.set(
                    '{:.2f}'.format(new_tg * 1000 / targ_tg))
                drowing.plot(screen, timeline, temp, mark_font,
                             energy_plot_box, 20, 20, 'К')
                drowing.plot(screen, timeline, mean_speed, mark_font,
                             speed_plot_box, 20, 20, 'м/с', new_tg, new_m)
            if timeline.shape[0] > 2000:
                tg_hist = np.append(tg_hist, [new_tg])
        pygame.draw.rect(screen, (240, 240, 240),
                         [0, 0, frame_model.winfo_width(),
                          frame_model.winfo_height()],
                         int(indent * 2.05))
        pygame.draw.rect(screen, (240, 240, 240),
                         [0, line_box[3][1], frame_model.winfo_width(),
                          2 * indent])
        pygame.draw.rect(screen, (240, 240, 240),
                         (indent, energy_plot_box[2][1],
                          energy_plot_box[1][0], 2 * indent))
        name = name_font.render('График температуры', 1, (28, 28, 28))
        place = name.get_rect(center=((energy_plot_box[0][0] +
                                       energy_plot_box[2][0]) // 2,
                                      (energy_plot_box[0][1] +
                                       model_box[1][1] - indent) // 2))
        screen.blit(name, place)
        name = name_font.render('График средней скорости частиц', 1,
                                (28, 28, 28))
        place = name.get_rect(center=((speed_plot_box[0][0] +
                                       speed_plot_box[2][0]) // 2,
                                      (energy_plot_box[2][1] +
                                       speed_plot_box[0][1]) // 2))
        screen.blit(name, place)
        root.update()


root = tk.Tk()
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(),
                                   root.winfo_screenheight()))

s = ttk.Style()
s.configure('my.TButton', font=('Helvetica', 18))

model_frame_height = 0.5
model_frame_width = 0.55

plot_flag = False
var_N = tk.IntVar(value=30)
N_lim = None
N = None
var_R = tk.DoubleVar(value=0.005)
R_lim = None
R = None
varlambda = None
var_dt_scale = tk.DoubleVar(value=1.0)
dt_scale = None
var_mass = tk.DoubleVar(value=0.018)
mass = 0.018
tg_hist = None
avogadro = float(6 * 10 ** 23)
mol_mass = mass / avogadro
Done = False
Model_is_run = False
t = 0
trust_index = 0.6
trust_lim = None
start_time = None
var_A = tk.DoubleVar(value=0.2)
A_lim = None
A = None
max_speed = None
var_rel = tk.DoubleVar(value=1.0)
rel = None
var_T = tk.DoubleVar(value=4.0)
T_lim = None
T = None
stabilized = None
var_initial_temp = tk.DoubleVar(value=10 ** (-9))
initial_temp = None
type_of_movement = tk.IntVar(value=1)
tg = None
var_tg = tk.DoubleVar(value=None)
targ_tg = None
var_targ_tg = tk.DoubleVar(value=None)
m = None
mean_dt = None
page = tk.IntVar(value=1)
n_page = None
learning_rate = 0.001
var_ratio_tg = tk.DoubleVar(value=None)

fps = 120
clock = pygame.time.Clock()

piston_coord = None
piston_speed = None

temp = None
mean_speed = None
timeline = None

coord = None
speed = None
mask = None
# Frames

work_frame = tk.Frame(root)
frame_model = tk.Frame(work_frame)
frame_param = tk.Frame(work_frame)
menu_frame = tk.Frame(root)
authors_frame = tk.Frame(root)
work_frame.place(relheight=1.0, relwidth=1.0)
frame_model.place(relheight=1.0, relwidth=model_frame_width)
frame_param.place(relx=model_frame_width, relheight=1.0,
                  relwidth=1 - model_frame_width)
menu_frame.place(relwidth=1.0, relheight=1.0)
authors_frame.place(relwidth=1.0, relheight=1.0)
frame_model.update()

# **Menu**

demo_button = ttk.Button(menu_frame, text='Демонстрация',
                         command=start_demonstration,
                         style='my.TButton')
demo_button.place(rely=0.7, relx=0.3, relheight=0.05, relwidth=0.4)
theor_button = ttk.Button(menu_frame, text='Теоретические обоснования',
                          command=theor,
                          style='my.TButton')
theor_button.place(rely=0.75, relx=0.3, relheight=0.05, relwidth=0.4)
authors_button = ttk.Button(menu_frame, text='Авторы',
                            command=authors,
                            style='my.TButton')
authors_button.place(rely=0.8, relx=0.3, relheight=0.05, relwidth=0.4)
exit_button = ttk.Button(menu_frame, text='Выход', command=end,
                         style='my.TButton')
exit_button.place(rely=0.85, relx=0.3, relheight=0.05, relwidth=0.4)
first_load = True
year = ttk.Label(menu_frame, text='2019', anchor='center',
                 font=(None, 16))
year.place(rely=0.92, relheight=0.05, relx=0.4, relwidth=0.2)

title = ttk.Label(menu_frame, text='Эффект ускорения\nФерми',
                  anchor='center', font=(None, 30), justify='center')
title.place(rely=0.4, relx=0.2, relheight=0.15, relwidth=0.6)
msu = ttk.Label(menu_frame, text='МГУ им. М.В. Ломоносова',
                anchor='center', font=(None, 20), justify='center')
msu.place(rely=0.05, relx=0.25, relheight=0.05, relwidth=0.5)
name = ttk.Label(menu_frame,
                 text='Компьютерные физические \nдемонстрации по'
                      ' курсу \nлекций \"Статистическая физика\"',
                 anchor='center', font=(None, 20), justify='center')
name.place(rely=0.15, relx=0.2, relheight=0.2, relwidth=0.6)
cmc_img = ImageTk.PhotoImage(
    Image.open('data/cmc.png').resize((130, 130), Image.ANTIALIAS))
cmc = ttk.Label(menu_frame, image=cmc_img, text='Факультет \nВМК',
                compound='top', font=(None, 20), justify='center')
cmc.place(rely=0.05, relx=0.25, x=-160, width=150, height=205)
ff_img = ImageTk.PhotoImage(
    Image.open('data/ff.png').resize((130, 130), Image.ANTIALIAS))
ff = ttk.Label(menu_frame, image=ff_img, text='Физический \nфакультет',
               compound='top', font=(None, 20), justify='center')
ff.place(rely=0.05, relx=0.75, x=10, width=150, height=205)

# **Authors**

exit_from_authors = ttk.Button(authors_frame, text='Меню',
                               command=go_to_menu, style='my.TButton')
exit_from_authors.place(rely=0.9, relx=0.8, relwidth=0.1,
                        relheight=0.05)
ruslan = ttk.Label(authors_frame, text='Руслан Шарыпов',
                   font=(None, 20), anchor='center')
alexandr = ttk.Label(authors_frame, text='Александр Деев',
                     font=(None, 20), anchor='center')
ruslan_photo = ImageTk.PhotoImage(Image.open('data/SR.jpg').resize((180, 240),
                                                    Image.ANTIALIAS))
ruslan_l = ttk.Label(authors_frame, image=ruslan_photo, justify='center')
ruslan_l.place(rely=0.1, y=200, relx=0.25, x=-90, width=180, height=240)
alexandr_photo = ImageTk.PhotoImage(Image.open('data/DA.jpg').resize((180, 240),
                                                    Image.ANTIALIAS))
alexandr_l = ttk.Label(authors_frame, image=alexandr_photo, justify='center')
alexandr_l.place(rely=0.1, y=200, relx=0.75, x=-90, width=180, height=240)
ruslan.place(relx=0.0, rely=0.65, relwidth=0.5, relheight=0.1)
alexandr.place(relx=0.5, rely=0.65, relwidth=0.5, relheight=0.1)
sensei = ttk.Label(authors_frame,
                   text='Преподаватель: Ольга Александровна Чичигина\n'
                        'Лектор: Анатолий Васильевич Андреев',
                   font=(None, 20), anchor='center', justify='center')
sensei.place(rely=0.75, relheight=0.1, relwidth=1.0)
auth_msu = ttk.Label(authors_frame, text='МГУ им. М.В. Ломоносова',
                     anchor='center', font=(None, 20), justify='center')
auth_msu.place(rely=0.05, relx=0.25, relheight=0.05, relwidth=0.5)
auth_name = ttk.Label(authors_frame,
                      text='Компьютерные физические \nдемонстрации по'
                           ' курсу \nлекций \"Статистическая физика\"',
                      anchor='center', font=(None, 20),
                      justify='center')
auth_name.place(rely=0.1, relx=0.2, relheight=0.2, relwidth=0.6)
auth_cmc = ttk.Label(authors_frame, image=cmc_img,
                     text='Факультет \nВМК',
                     compound='top', font=(None, 20), justify='center')
auth_cmc.place(rely=0.05, relx=0.25, x=-160, width=150, height=205)
auth_ff = ttk.Label(authors_frame, image=ff_img,
                    text='Физический \nфакультет',
                    compound='top', font=(None, 20), justify='center')
auth_ff.place(rely=0.05, relx=0.75, x=10, width=150, height=205)

# **Control buttons**

menu_button = ttk.Button(frame_param, text='Меню', command=go_to_menu,
                         style='my.TButton')
menu_button.place(relx=0.7, relwidth=0.25, rely=1.0, y=-80, height=60)
back_button = ttk.Button(frame_param, text ='<', command=left,
                         style='my.TButton')
back_button.place(relx=0.05, relwidth=0.15, rely=1.0, y=-80, height=60)
forward_button = ttk.Button(frame_param, text='>', command=right,
                            style='my.TButton')
forward_button.place(relx=0.5, relwidth=0.15, rely=1.0, y=-80, height=60)
page_label = ttk.Label(frame_param, text='Стр.', compound='right',
                       style='NameVar.TLabel')
page_label.place(relx=0.25, relwidth=0.1, rely=1.0, y=-80, height=60)
page_num_label = ttk.Label(frame_param, textvariable=page,
                       style='NameVar.TLabel')
page_num_label.place(relx=0.35, relwidth=0.1, rely=1.0, y=-80, height=60)
play_pause_button = ttk.Button(frame_param, text='Старт',
                               command=play_pause, style='my.TButton')
play_pause_button.place(relx=0.05, relwidth=0.425, y=30, height=60)
restart_button = ttk.Button(frame_param, text='Перезапуск',
                            command=restart, style='my.TButton')
restart_button.place(relx=0.525, relwidth=0.425, y=30, height=60)

var_style = ttk.Style()
var_style.configure('Var.TLabel', background='white',
                    font=('Helvetica', 18))
var_style.configure('NameVar.TLabel', font=('Helvetica', 16))
var_style.configure('My.TRadiobutton', font=('Helvetica', 16))

# **Params_control**

pages = [tk.Frame(frame_param), tk.Frame(frame_param)]
for i in pages:
    i.place(relwidth=1.0, relheight=0.7, height=-170, y=90)

pages[0].tkraise()
n_page = len(pages)

column_width = [0.45, 0.2, 0.15]
column_offset = [0.05, 0.55, 0.8]
row_height = 0.2
row_offset = 0.04
base_row_offset = 0.04
max_amplitude = int(root.winfo_width() * model_frame_width / 2)


# Type of piston movement


def set_rel(*args):
    global var_rel, rel_scale
    var_rel.set(1.0)
    if type_of_movement.get() == 0:
        rel_scale['state'] = 'disabled'
    else:
        rel_scale['state'] = 'normal'


type_of_piston_movement_label = ttk.Label(pages[0],
                                          text='Характер\nдвижения\n '
                                               'поршня', justify='left',
                                          style='NameVar.TLabel')
sin_type = ttk.Radiobutton(pages[0], text='Гармонический',
                           variable=type_of_movement, value=0,
                           command=set_rel, style='My.TRadiobutton')
linear_type = ttk.Radiobutton(pages[0], text='Линейный',
                              variable=type_of_movement, value=1,
                              command=set_rel, style='My.TRadiobutton')
type_of_piston_movement_label.place(relx=0.05, relwidth=0.25,
                                    rely=row_offset, relheight=row_height)
sin_type.place(relx=0.3, relwidth=0.4, rely=row_offset,
               relheight=row_height)
linear_type.place(relx=0.7, relwidth=0.3,
                  rely=row_offset, relheight=row_height)
row_offset += base_row_offset + row_height

# Amplitude


A_lim = [0.1, 0.2]
amplitude_spin = ttk.Label(pages[0], textvariable=var_A,
                           style='Var.TLabel', anchor='center')
amplitude_scale = tk.Scale(pages[0], from_=A_lim[0],
                           to=A_lim[1], variable=var_A, digits=2,
                           resolution=0.01, showvalue=0,
                           orient=tk.HORIZONTAL)
amplitude_label = ttk.Label(pages[0], text='Амплитуда:',
                            style='NameVar.TLabel')
amplitude_label.place(relx=column_offset[0], relwidth=column_width[0],
                      rely=row_offset, relheight=row_height)
amplitude_spin.place(relx=column_offset[1], relwidth=column_width[1],
                     rely=row_offset, relheight=row_height)
amplitude_scale.place(relx=column_offset[2], relwidth=column_width[2],
                      rely=row_offset, relheight=row_height)
row_offset += base_row_offset + row_height

# Rel_per


rel_lim = [0.5, 2]
rel_spin = ttk.Label(pages[0], textvariable=var_rel,
                     style='Var.TLabel', anchor='center')
rel_scale = tk.Scale(pages[0], from_=rel_lim[0], to=rel_lim[1],
                     variable=var_rel, digits=2,
                     resolution=0.01, showvalue=0,
                     orient=tk.HORIZONTAL)
rel_label = ttk.Label(pages[0],
                      text='Отношение\nвремени\nсжат. и расш.:',
                      style='NameVar.TLabel')
rel_label.place(relx=column_offset[0], relwidth=column_width[0],
                rely=row_offset, relheight=row_height)
rel_spin.place(relx=column_offset[1], relwidth=column_width[1],
               rely=row_offset, relheight=row_height)
rel_scale.place(relx=column_offset[2], relwidth=column_width[2],
                rely=row_offset, relheight=row_height)
row_offset += base_row_offset + row_height

# Period


T_lim = [3, 10]
period_spin = ttk.Label(pages[0], textvariable=var_T,
                        style='Var.TLabel', anchor='center')
period_scale = tk.Scale(pages[0], from_=T_lim[0], to=T_lim[1],
                        variable=var_T, digits=3,
                        resolution=0.1, showvalue=0,
                        orient=tk.HORIZONTAL)
period_label = ttk.Label(pages[0], text='Период, сек.:',
                         style='NameVar.TLabel')
period_label.place(relx=column_offset[0], relwidth=column_width[0],
                   rely=row_offset, relheight=row_height)
period_spin.place(relx=column_offset[1], relwidth=column_width[1],
                  rely=row_offset, relheight=row_height)
period_scale.place(relx=column_offset[2], relwidth=column_width[2],
                   rely=row_offset, relheight=row_height)
row_offset += base_row_offset + row_height

# Number of molecules
row_offset = base_row_offset

N_lim = [1, 80]
number_of_molecules_spin = ttk.Label(pages[1], textvariable=var_N,
                                     style='Var.TLabel',
                                     anchor='center')
number_of_molecules_scale = tk.Scale(pages[1], from_=N_lim[0],
                                     to=N_lim[1], variable=var_N,
                                     digits=0,
                                     resolution=1, showvalue=0,
                                     orient=tk.HORIZONTAL)
number_of_molecules_label = ttk.Label(pages[1],
                                      text='Количество молекул:',
                                      style='NameVar.TLabel')
number_of_molecules_label.place(relx=column_offset[0],
                                relwidth=column_width[0],
                                rely=row_offset,
                                relheight=row_height)
number_of_molecules_spin.place(relx=column_offset[1],
                               relwidth=column_width[1],
                               rely=row_offset,
                               relheight=row_height)
number_of_molecules_scale.place(relx=column_offset[2],
                                relwidth=column_width[2],
                                rely=row_offset,
                                relheight=row_height)
row_offset += base_row_offset + row_height

# Molecule radius


R_lim = [0.005, 0.02]
molecule_radius_spin = ttk.Label(pages[1], textvariable=var_R,
                                 style='Var.TLabel', anchor='center')
molecule_radius_scale = tk.Scale(pages[1], from_=R_lim[0],
                                 to=R_lim[1], variable=var_R, digits=2,
                                 resolution=0.001, showvalue=0,
                                 orient=tk.HORIZONTAL)
molecule_radius_label = ttk.Label(pages[1],
                                  text='Радиус молекул:',
                                  style='NameVar.TLabel')
molecule_radius_label.place(relx=column_offset[0],
                            relwidth=column_width[0], rely=row_offset,
                            relheight=row_height)
molecule_radius_spin.place(relx=column_offset[1],
                           relwidth=column_width[1], rely=row_offset,
                           relheight=row_height)
molecule_radius_scale.place(relx=column_offset[2],
                            relwidth=column_width[2], rely=row_offset,
                            relheight=row_height)
row_offset += base_row_offset + row_height

# Molecular mass


mass_lim = [0.001, 0.5]

molecule_mass_spin = ttk.Label(pages[1], textvariable=var_mass,
                               style='Var.TLabel', anchor='center')
molecule_mass_scale = tk.Scale(pages[1], from_=mass_lim[0],
                               to=mass_lim[1], variable=var_mass,
                               digits=3,
                               resolution=0.001, showvalue=0,
                               orient=tk.HORIZONTAL)
molecule_mass_label = ttk.Label(pages[1],
                                text='Молекулярная\nмасса, кг/моль:',
                                style='NameVar.TLabel')
molecule_mass_label.place(relx=column_offset[0],
                          relwidth=column_width[0], rely=row_offset,
                          relheight=row_height)
molecule_mass_spin.place(relx=column_offset[1],
                         relwidth=column_width[1], rely=row_offset,
                         relheight=row_height)
molecule_mass_scale.place(relx=column_offset[2],
                          relwidth=column_width[2], rely=row_offset,
                          relheight=row_height)
row_offset += base_row_offset + row_height


# Initial temperature


def exp(*args):
    global var_initial_temp
    var_initial_temp.set("{:.2g}".format(float(var_initial_temp.get())))


var_initial_temp.trace_variable('r', exp)
initial_temp_lim = [10 ** (-10), 10 ** (-4)]
initial_temp_spin = ttk.Label(pages[1],
                              textvariable=var_initial_temp,
                              style='Var.TLabel', anchor='center')
initial_temp_scale = tk.Scale(pages[1],
                              from_=initial_temp_lim[0],
                              to=initial_temp_lim[1],
                              variable=var_initial_temp, command=exp,
                              digits=2,
                              resolution=10 ** (-10), showvalue=0,
                              orient=tk.HORIZONTAL)
initial_temp_label = ttk.Label(pages[1],
                               text='Начальная\nтемпература, К:',
                               style='NameVar.TLabel', justify='center')
initial_temp_label.place(relx=column_offset[0],
                         relwidth=column_width[0], rely=row_offset,
                         relheight=row_height)
initial_temp_spin.place(relx=column_offset[1],
                        relwidth=column_width[1], rely=row_offset,
                        relheight=row_height)
initial_temp_scale.place(relx=column_offset[2],
                         relwidth=column_width[2], rely=row_offset,
                         relheight=row_height)
row_offset += base_row_offset + row_height

# Culc speed
dt_scale_lim = [0.01, 1.0]
dt_scale_label = ttk.Label(frame_param,
                           text='Скорость демонстрации:',
                           style='NameVar.TLabel')
dt_scale_scale = ttk.Scale(frame_param,
                           from_=dt_scale_lim[0],
                           to=dt_scale_lim[1],
                           variable=var_dt_scale)
dt_scale_label.place(relx=column_offset[0],
                     relwidth=column_width[0] + column_width[1], y=-80,
                     rely=0.7,
                     relheight=0.05)
dt_scale_scale.place(relx=column_offset[2],
                     relwidth=column_width[2], y=-90, rely=0.7,
                     relheight=0.05)
row_offset += base_row_offset + row_height

# Tg
tg_label = ttk.Label(frame_param,
                     text='Показатель наклона графика скорости',
                     style='NameVar.TLabel', anchor='center')
ratio_tg_show = ttk.Label(frame_param, textvariable=var_ratio_tg,
                          style='Var.TLabel', anchor='center')
ratio_tg_label = ttk.Label(frame_param, text='Отношение',
                           style='NameVar.TLabel', anchor='center')
tg_show = ttk.Label(frame_param, textvariable=var_tg,
                    style='Var.TLabel', anchor='center')
culc_tg_label = ttk.Label(frame_param, text='Эксперемент',
                          style='NameVar.TLabel', anchor='center')
targ_tg_show = ttk.Label(frame_param, textvariable=var_targ_tg,
                         style='Var.TLabel', anchor='center')
targ_tg_label = ttk.Label(frame_param, text='Теория (бильярд)',
                          style='NameVar.TLabel', anchor='center')
tg_label.place(relx=0.05, relwidth=0.9,
               y=-90, rely=0.8, relheight=0.05)
culc_tg_label.place(relx=0.05, relwidth=0.3, y=-90, rely=0.85,
                    relheight=0.05)
ratio_tg_label.place(relx=0.4, relwidth=0.2, y=-90, rely=0.85,
                     relheight=0.05)
targ_tg_label.place(relx=0.65, relwidth=0.3, y=-90, rely=0.85,
                    relheight=0.05)
tg_show.place(relx=0.05, relwidth=0.3, y=-90, rely=0.9, relheight=0.05)
ratio_tg_show.place(relx=0.4, relwidth=0.2, y=-90, rely=0.9,
                    relheight=0.05)
targ_tg_show.place(relx=0.65, relwidth=0.3, y=-90, rely=0.9,
                   relheight=0.05)

os.environ['SDL_WINDOWID'] = str(frame_model.winfo_id())
os.environ['SDL_VIDEODRIVER'] = 'windib'

screen = pygame.display.set_mode((frame_model.winfo_width(),
                                  frame_model.winfo_height()))

pygame.display.init()
pygame.font.init()
pygame.display.update()

mark_font = pygame.font.SysFont('Sans', 21)
name_font = pygame.font.SysFont('Sans', 31)
model_box = [[0, 0],
             [frame_model.winfo_width(),
              int(frame_model.winfo_height() * model_frame_height)]]
indent = 20
line_box = [[model_box[0][0] + indent, model_box[0][1] + indent],
            [model_box[1][0] - indent, model_box[0][1] + indent],
            [model_box[1][0] - indent, model_box[1][1] - indent],
            [model_box[0][0] + indent, model_box[1][1] - indent]]
piston_box = [indent, indent, 0, model_box[1][1] - 2 * indent]
bord_box = np.array(model_box)
bord_box[0] += int(indent * 1.5)
bord_box[1] -= int(indent * 1.5)
plots_height = int(frame_model.winfo_height() *
                   (1 - model_frame_height) * 0.5)
energy_plot_box = [[indent, model_box[1][1] + indent],
                   [model_box[1][0] - 2 * indent,
                    plots_height - 2 * indent],
                   [frame_model.winfo_width() - indent,
                    frame_model.winfo_height() - plots_height - indent]]

speed_plot_box = [[indent, frame_model.winfo_height() -
                   plots_height + indent],
                  [model_box[1][0] - 2 * indent,
                   plots_height - 2 * indent],
                  [frame_model.winfo_width() - indent,
                   frame_model.winfo_height() - indent]]

size_coef = np.mean([frame_model.winfo_screenmmwidth() /
                     frame_model.winfo_screenwidth(),
                     frame_model.winfo_screenmmheight() /
                     frame_model.winfo_screenheight()]) / 1000

menu_frame.tkraise()
while not Done:
    pygame.time.delay(10)
    root.update()
    if root.focus_get() != None:
        root.overrideredirect(True)
    else:
        root.overrideredirect(False)
        root.iconify()
    pygame.display.update()

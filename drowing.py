import numpy as np
import pygame


def f(x, screen, radius):
    pygame.draw.circle(screen, (48, 128, 20), x.astype(int), radius)


draw_mol = np.vectorize(f, signature='(n)->()',
                        excluded=['screen', 'radius'])


def draw_sys(coord, screen, line_box, piston_box, thickness, radius,
             fps, mark_font, timeline):
    screen.fill((255, 255, 255))
    fps_mark = mark_font.render('{:.2f}fps, shape: {:d}'.format(fps, timeline), 1,
                                (255, 99, 71))
    pygame.draw.rect(screen, (112, 128, 144), piston_box)
    pygame.draw.lines(screen, (181, 181, 181), False, line_box,
                      thickness)
    draw_mol(coord, screen=screen, radius=radius)
    #screen.blit(fps_mark, (line_box[0][0] + 10, line_box[0][1] + 10))


def plot(screen, x, y, mark_font, plot_box, mark_height, mark_width,
         unit, tg=None, m=None):
    pygame.draw.rect(screen, (255, 255, 255), plot_box[:-1])
    pygame.draw.rect(screen, (181, 181, 181), plot_box[:-1], 5)
    x_scale_point = 10 ** (np.ceil(np.log10(x[-1])) - 1)
    x_scale = np.arange(1,
                        int(x[-1] / x_scale_point) + 1) * x_scale_point
    for point in x_scale:
        x_mark = mark_font.render('{:.0e} с'.format(point), 1,
                                  (82, 82, 82))
        screen.blit(x_mark, (point / x[-1] * plot_box[1][0] +
                             plot_box[0][0],
                             plot_box[2][1] - mark_height))
        pygame.draw.line(screen, (220, 220, 220),
                         [point / x[-1] * plot_box[1][0] +
                          plot_box[0][0], plot_box[0][1]],
                         [point / x[-1] * plot_box[1][0] +
                          plot_box[0][0], plot_box[2][1]])
    y_max = np.max(y)
    y_scale_point = 10 ** (np.ceil(np.log10(y_max)) - 1)
    y_scale = np.arange(1,
                        int(y_max / y_scale_point) + 1) * y_scale_point
    for point in y_scale:
        y_mark = mark_font.render('{:.1e} '.format(point) + unit, 1,
                                  (82, 82, 82))
        screen.blit(y_mark, (plot_box[0][0] + mark_width,
                             - point / y_max * plot_box[1][1]
                             + plot_box[2][1]))
        pygame.draw.line(screen, (220, 220, 220),
                         [plot_box[0][0],
                          - point / y_max * plot_box[1][1]
                          + plot_box[2][1]],
                         [plot_box[2][0],
                          - point / y_max * plot_box[1][1]
                          + plot_box[2][1]], )
    if tg != None:
        approximating_line = np.array([x[0], x[-1]]) * tg + m
        approximating_line = - approximating_line / y_max * \
                             plot_box[1][1] + plot_box[2][1]
    x_norm = x / x[-1] * plot_box[1][0] + plot_box[0][0]
    y_norm = - y / y_max * plot_box[1][1] + plot_box[2][1]
    pygame.draw.lines(screen, (0, 0, 255), False,
                      np.stack((x_norm, y_norm), axis=-1), 2)
    if tg != None:
        pygame.draw.line(screen, (255, 0, 0), (x_norm[0], approximating_line[0]), (x_norm[-1], approximating_line[1]), 5)

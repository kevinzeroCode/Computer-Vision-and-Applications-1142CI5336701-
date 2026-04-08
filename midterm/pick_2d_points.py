"""
pick_2d_points.py
Interactive tool to pick 2D pixel coordinates from an image.
Usage:
    python pick_2d_points.py --img "7Images and xyz/01.jpg" --out points/img01_2d.txt [--n 10]

Controls:
    Left click  → add point
    Right click → remove last point
    Enter / q   → save and exit
    z           → undo last point
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # use interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Path to image file')
    parser.add_argument('--out', required=True, help='Output txt file for 2D points')
    parser.add_argument('--n', type=int, default=0, help='Expected number of points (0 = unlimited)')
    parser.add_argument('--ref', default='', help='Optional: 3D points file to show landmark labels')
    args = parser.parse_args()

    img = np.array(Image.open(args.img))
    h, w = img.shape[:2]

    labels = []
    if args.ref:
        try:
            data = np.loadtxt(args.ref)
            labels = [f"P{i+1}: ({row[0]:.2f},{row[1]:.2f},{row[2]:.2f})"
                      for i, row in enumerate(data)]
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    title_base = f'{args.img}  |  Click points in order. Enter/q to save.'
    ax.set_title(title_base)

    points = []          # list of (u, v)
    scatter_artists = []
    text_artists = []

    def redraw():
        for a in scatter_artists:
            a.remove()
        for a in text_artists:
            a.remove()
        scatter_artists.clear()
        text_artists.clear()
        for i, (u, v) in enumerate(points):
            s = ax.plot(u, v, 'r+', markersize=12, markeredgewidth=2)[0]
            scatter_artists.append(s)
            label = labels[i] if i < len(labels) else f'P{i+1}'
            t = ax.text(u + 8, v - 8, label, color='yellow', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
            text_artists.append(t)
        n_msg = f'  [{len(points)}/{args.n}]' if args.n else f'  [{len(points)} pts]'
        ax.set_title(title_base + n_msg)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # left click → add
            if args.n and len(points) >= args.n:
                print(f'Already have {args.n} points. Press Enter/q to save.')
                return
            u, v = event.xdata, event.ydata
            points.append((u, v))
            print(f'  Point {len(points)}: u={u:.1f}, v={v:.1f}')
            redraw()
        elif event.button == 3:  # right click → undo
            if points:
                points.pop()
                print(f'  Removed last point. Now {len(points)} points.')
                redraw()

    def on_key(event):
        if event.key in ('enter', 'q'):
            save_and_exit()
        elif event.key == 'z':
            if points:
                points.pop()
                print(f'  Undo. Now {len(points)} points.')
                redraw()

    def save_and_exit():
        if len(points) == 0:
            print('No points selected. Exiting without saving.')
            plt.close()
            return
        arr = np.array(points)
        np.savetxt(args.out, arr, fmt='%.4f', header='u v')
        print(f'\nSaved {len(points)} points to {args.out}')
        plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

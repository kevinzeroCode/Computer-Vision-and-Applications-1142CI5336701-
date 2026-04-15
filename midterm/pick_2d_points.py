"""
pick_2d_points.py
Interactive tool to pick 2D pixel coordinates from an image.

Usage (basic):
    python pick_2d_points.py --img "7Images and xyz/01.jpg" --out points/img01_2d.txt

Usage (structured checklist — recommended):
    python pick_2d_points.py --img "7Images and xyz/01.jpg" --out points/img01_2d.txt \
        --names points/landmark_names.txt --checklist

Usage (fix mode — re-pick only specific landmarks, keep the rest):
    python pick_2d_points.py --img "7Images and xyz/02.jpg" --out points/img02_2d.txt \
        --checklist --fix belt_bump_L3,belt_bump_L4

Controls:
    Left click        → add point (or confirm selected landmark in checklist mode)
    Right click / z   → undo last point
    Enter / q         → save and exit
    ↑ / ↓ arrow keys  → move checklist selection (checklist mode only)
    Space             → skip current landmark (mark as not visible)

Improvements in this version:
    A1  Live (u, v) coordinate display in title bar while moving mouse
    A2  Magnifier loupe in bottom-right corner (4× zoom, 80×80 px window)
    A4  Crosshair lines following the cursor
    B2  Checklist shows recommended images per landmark
    C1  Fix mode: pre-load existing file, only re-pick specified landmarks
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


LANDMARK_NAMES = [
    'hat_tip',
    'nose',
    'right_shoulder', 'left_shoulder',
    'welcome_top_R', 'welcome_top_L',
    'welcome_bot_R', 'welcome_bot_L',
    'foot_R', 'foot_L',
    'belt_welcome_R', 'belt_welcome_L',
    'right_eye_inner', 'right_eye_outer',
    'left_eye_inner',  'left_eye_outer',
    'hat_brim_front', 'hat_brim_R', 'hat_brim_L',
    'belt_bump_R1', 'belt_bump_R2', 'belt_bump_R3', 'belt_bump_R4',
    'belt_bump_L1', 'belt_bump_L2', 'belt_bump_L3', 'belt_bump_L4',
    'mustache_R', 'mustache_L', 'beard_tip',
    'shoe_heel_R', 'trouser_hem_R',
    'shoe_heel_L', 'trouser_hem_L',
]

# Camera centers: img01=(84,-2,53) right-side; img02=(-83,1,65) left-side;
#   img03=(72,112,41) right-front; img04=(-91,-91,-10) left-back-low;
#   img05=(-9,131,30) front; img06=(-70,-58,40) left-back; img07=(41,-76,32) right-back
LANDMARK_VISIBILITY = {
    'hat_tip':          '1-7',
    'nose':             '3,5',
    'right_shoulder':   '1,3,5,7',
    'left_shoulder':    '2,4,5,6',
    'welcome_top_R':    '3,5',
    'welcome_top_L':    '3,5',
    'welcome_bot_R':    '3,5',
    'welcome_bot_L':    '3,5',
    'foot_R':           '1,3,5,7',
    'foot_L':           '2,4,5,6',
    'belt_welcome_R':   '3,5',
    'belt_welcome_L':   '3,5',
    'right_eye_inner':  '1,3,5',
    'right_eye_outer':  '1,3,5,7',
    'left_eye_inner':   '2,3,5',
    'left_eye_outer':   '2,4,5,6',
    'hat_brim_front':   '3,5',
    'hat_brim_R':       '1,3,5,7',
    'hat_brim_L':       '2,4,5,6',
    'belt_bump_R1':     '1,3,5',      # front side of right belt
    'belt_bump_R2':     '1,3,5',
    'belt_bump_R3':     '1,7',        # back side of right belt
    'belt_bump_R4':     '1,7',
    'belt_bump_L1':     '2,4,5',      # front side of left belt
    'belt_bump_L2':     '2,4,5',
    'belt_bump_L3':     '2,6',        # back side of left belt
    'belt_bump_L4':     '2,6',
    'mustache_R':       '1,3,5',
    'mustache_L':       '2,3,5',
    'beard_tip':        '3,5',
    'shoe_heel_R':      '1,7',        # right shoe back — right-side & right-back
    'trouser_hem_R':    '1,7',        # right trouser hem — right-side & right-back
    'shoe_heel_L':      '2,4,6',      # left shoe back — left-side, left-back-low, left-back
    'trouser_hem_L':    '2,4,6',      # left trouser hem
}

# Magnifier parameters
LOUPE_HALF   = 40   # half-size of source window (pixels in original image)
LOUPE_ZOOM   = 4    # display zoom factor
LOUPE_SIZE   = LOUPE_HALF * 2 * LOUPE_ZOOM   # output size in display pixels


def load_names(path):
    names = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    names.append(line)
    except FileNotFoundError:
        pass
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',       required=True, help='Path to image file')
    parser.add_argument('--out',       required=True, help='Output txt file for 2D points')
    parser.add_argument('--n',         type=int, default=0, help='Expected number of points (0=unlimited)')
    parser.add_argument('--ref',       default='', help='3D points file (for backward compat)')
    parser.add_argument('--names',     default='', help='File with one landmark name per line')
    parser.add_argument('--checklist', action='store_true',
                        help='Enable structured checklist mode (pick landmarks by name)')
    parser.add_argument('--fix',       default='',
                        help='Fix mode: comma-separated landmark names to re-pick. '
                             'All other points are loaded from --out and locked.')
    args = parser.parse_args()

    img = np.array(Image.open(args.img))

    # Load landmark names
    if args.names:
        names = load_names(args.names)
    else:
        names = list(LANDMARK_NAMES)

    # ── Fix mode: load existing file, restrict to specified landmarks ─────── #
    fix_names = set()
    preloaded = {}   # name → (u, v)  — locked points from existing file
    if args.fix:
        fix_names = set(n.strip() for n in args.fix.split(',') if n.strip())
        # Load existing annotations from --out if it exists
        import os
        if os.path.exists(args.out):
            try:
                with open(args.out, encoding='utf-8') as fh:
                    header = fh.readline().strip().lstrip('#').split()
                pts = np.loadtxt(args.out, comments='#')
                if pts.ndim == 1:
                    pts = pts[None, :]
                for nm, row in zip(header, pts):
                    preloaded[nm] = (float(row[0]), float(row[1]))
                print(f'Fix mode: loaded {len(preloaded)} existing pts, will re-pick: {fix_names}')
            except Exception as e:
                print(f'Warning: could not load existing file: {e}')
        # Filter names to only those in the existing file + fix targets
        existing_names = list(preloaded.keys())
        # Keep order: existing names first, add any new fix_names not yet present
        ordered = existing_names[:]
        for fn in fix_names:
            if fn not in ordered:
                ordered.append(fn)
        names = ordered

    # ── Checklist mode ────────────────────────────────────────────────────── #
    if args.checklist:
        _run_checklist(img, names, args.out, args.img,
                       fix_names=fix_names, preloaded=preloaded)
        return

    # ── Classic mode (original behaviour + A1/A2/A4) ─────────────────────── #
    labels = []
    if args.ref:
        try:
            data = np.loadtxt(args.ref)
            labels = [f"P{i+1}: ({row[0]:.2f},{row[1]:.2f},{row[2]:.2f})"
                      for i, row in enumerate(data)]
        except Exception:
            pass
    elif names:
        labels = names

    img_h, img_w = img.shape[:2]
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#111111')
    # Image occupies left 72%; right 28% is the sidebar
    ax       = fig.add_axes([0.0,  0.0, 0.72, 1.0])
    ax_side  = fig.add_axes([0.73, 0.0, 0.26, 1.0])
    ax_side.set_facecolor('#111111')
    ax_side.set_axis_off()
    ax.imshow(img)
    title_base = f'{args.img}  |  Left-click add · Right-click/z undo · Enter/q save'
    ax.set_title(title_base, fontsize=9, color='white')

    # ── A2: Magnifier in right sidebar (no image overlap) ────────────────── #
    ax_loupe = fig.add_axes([0.74, 0.58, 0.24, 0.36])
    ax_loupe.set_axis_off()
    loupe_im = ax_loupe.imshow(np.zeros((LOUPE_HALF*2, LOUPE_HALF*2, 3), dtype=np.uint8))
    ax_loupe.set_title('Magnifier  (4×)', color='#aaffaa', fontsize=8, pad=3)
    ax_loupe.axhline(LOUPE_HALF - 0.5, color='red', lw=0.8, alpha=0.7)
    ax_loupe.axvline(LOUPE_HALF - 0.5, color='red', lw=0.8, alpha=0.7)
    for spine in ax_loupe.spines.values():
        spine.set_edgecolor('#44ff44')
        spine.set_linewidth(1.2)
        spine.set_visible(True)
    # Coord display text in sidebar
    coord_txt = ax_side.text(0.5, 0.50, 'u=—  v=—',
                             ha='center', va='center', color='#ffff88',
                             fontsize=10, transform=ax_side.transAxes)

    # ── A4: Crosshair lines ───────────────────────────────────────────────── #
    ch_h = ax.axhline(-1, color='cyan', lw=0.6, alpha=0.5, ls='--')
    ch_v = ax.axvline(-1, color='cyan', lw=0.6, alpha=0.5, ls='--')

    points, scat, texts = [], [], []

    def _update_loupe(u, v):
        """Refresh magnifier with 4× zoom centred on (u, v)."""
        u_i, v_i = int(round(u)), int(round(v))
        y0 = max(v_i - LOUPE_HALF, 0)
        y1 = min(v_i + LOUPE_HALF, img_h)
        x0 = max(u_i - LOUPE_HALF, 0)
        x1 = min(u_i + LOUPE_HALF, img_w)
        patch = img[y0:y1, x0:x1]
        # pad if near border
        pad_top    = max(LOUPE_HALF - v_i, 0)
        pad_bottom = max(v_i + LOUPE_HALF - img_h, 0)
        pad_left   = max(LOUPE_HALF - u_i, 0)
        pad_right  = max(u_i + LOUPE_HALF - img_w, 0)
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            patch = np.pad(patch,
                           ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                           mode='constant', constant_values=30)
        loupe_im.set_data(patch)
        loupe_im.set_extent([-0.5, patch.shape[1]-0.5, patch.shape[0]-0.5, -0.5])

    def redraw():
        for a in scat + texts:
            a.remove()
        scat.clear(); texts.clear()
        for i, (u, v) in enumerate(points):
            scat.append(ax.plot(u, v, 'r+', ms=12, mew=2)[0])
            lbl = labels[i] if i < len(labels) else f'P{i+1}'
            texts.append(ax.text(u+8, v-8, lbl, color='yellow', fontsize=8,
                                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55)))
        msg = f'  [{len(points)}/{args.n}]' if args.n else f'  [{len(points)} pts]'
        ax.set_title(title_base + msg, fontsize=9, color='white')
        fig.canvas.draw_idle()

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        u, v = event.xdata, event.ydata
        # A1: live coords in sidebar text
        coord_txt.set_text(f'u = {u:.1f}\nv = {v:.1f}')
        # A4: crosshair
        ch_h.set_ydata([v, v])
        ch_v.set_xdata([u, u])
        # A2: loupe
        _update_loupe(u, v)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax: return
        if event.button == 1:
            if args.n and len(points) >= args.n:
                print(f'Already at {args.n} points. Press Enter/q to save.')
                return
            points.append((event.xdata, event.ydata))
            print(f'  P{len(points)}: u={event.xdata:.1f}  v={event.ydata:.1f}')
            redraw()
        elif event.button == 3:
            if points:
                points.pop(); redraw()
                print(f'  Undo. {len(points)} pts remaining.')

    def on_key(event):
        if event.key in ('enter', 'q'):
            _save(points, args.out); plt.close()
        elif event.key == 'z':
            if points:
                points.pop(); redraw()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event',  on_click)
    fig.canvas.mpl_connect('key_press_event',      on_key)
    plt.tight_layout()
    plt.show()


# ── Checklist mode implementation ─────────────────────────────────────────── #

def _run_checklist(img, names, out_path, img_title,
                   fix_names=None, preloaded=None):
    """
    Split view: image (left) + landmark checklist (right).
    Arrow keys move selection; left-click on image picks that landmark;
    Space skips (marks invisible); Enter/q saves.

    fix_names  : set of landmark names to re-pick (others are locked)
    preloaded  : dict name→(u,v) of existing locked points

    Additions vs original:
      A1  live (u,v) in title while moving
      A2  magnifier inset in bottom-right of image pane
      A4  crosshair following cursor
      B2  checklist shows recommended images per landmark
      C1  fix mode: locked points shown in grey, only fix_names active
    """
    fix_names = fix_names or set()
    preloaded = preloaded or {}

    img_h, img_w = img.shape[:2]
    n_lm = len(names)
    # Pre-populate locked (non-fix) points into picked
    picked   = {}    # name → (u, v)
    locked   = set() # names that cannot be changed
    skipped  = set()
    for nm, uv in preloaded.items():
        if nm in names:
            if nm not in fix_names:
                picked[nm] = uv
                locked.add(nm)
    # sel_idx starts at first fix landmark (or 0)
    sel_idx = 0
    if fix_names:
        for i, nm in enumerate(names):
            if nm in fix_names:
                sel_idx = i
                break

    fig = plt.figure(figsize=(17, 9))
    fig.patch.set_facecolor('#1a1a2e')
    # Image: left 66%; right panel split: checklist top 55%, magnifier bottom 38%
    ax_img  = fig.add_axes([0.0,  0.0,  0.66, 1.0])
    ax_list = fig.add_axes([0.67, 0.38, 0.32, 0.62])
    ax_img.imshow(img)
    ax_img.set_axis_off()
    ax_list.set_axis_off()
    ax_list.set_xlim(0, 1); ax_list.set_ylim(0, 1)

    # ── A2: Magnifier in right panel bottom — no image overlap ───────────── #
    ax_loupe = fig.add_axes([0.67, 0.01, 0.32, 0.34])
    ax_loupe.set_axis_off()
    loupe_im = ax_loupe.imshow(np.zeros((LOUPE_HALF*2, LOUPE_HALF*2, 3), dtype=np.uint8))
    ax_loupe.set_title('Magnifier  4×', color='#aaffaa', fontsize=8, pad=3)
    ax_loupe.axhline(LOUPE_HALF - 0.5, color='red', lw=0.8, alpha=0.7)
    ax_loupe.axvline(LOUPE_HALF - 0.5, color='red', lw=0.8, alpha=0.7)
    for spine in ax_loupe.spines.values():
        spine.set_edgecolor('#44ff44'); spine.set_linewidth(1.2); spine.set_visible(True)

    # ── A4: Crosshair ─────────────────────────────────────────────────────── #
    ch_h = ax_img.axhline(-1, color='cyan', lw=0.6, alpha=0.45, ls='--')
    ch_v = ax_img.axvline(-1, color='cyan', lw=0.6, alpha=0.45, ls='--')

    title_base = (f'{img_title}\n'
                  '↑↓ select · Left-click pick · Space skip · z undo · L labels · Enter/q save')

    # A1: live coord text at top of right panel (above checklist)
    coord_txt = ax_list.text(0.5, 1.06, 'u=—  v=—',
                             ha='center', va='bottom', color='#ffff88',
                             fontsize=9, transform=ax_list.transAxes)

    scat_artists  = {}
    label_artists = {}
    list_texts    = {}
    list_bg       = {}
    vis_texts     = {}   # B2: visibility hint text per row
    show_labels   = [True]   # mutable flag for label toggle

    row_h = 0.90 / max(n_lm, 1)

    def _row_y(i):
        return 0.95 - i * row_h

    ax_list.text(0.5, 0.985, 'Landmark Checklist',
                 ha='center', va='top', color='white',
                 fontsize=9, fontweight='bold',
                 transform=ax_list.transAxes)

    for i, name in enumerate(names):
        y = _row_y(i)
        bg = FancyBboxPatch((0.01, y - row_h*0.45), 0.98, row_h*0.85,
                            boxstyle='round,pad=0.01',
                            fc='#2a2a4a', ec='none',
                            transform=ax_list.transAxes, clip_on=False)
        ax_list.add_patch(bg)
        list_bg[i] = bg

        t = ax_list.text(0.06, y + row_h*0.12, f'○  {name}',
                         va='center', color='#aaaacc', fontsize=7.5,
                         transform=ax_list.transAxes)
        list_texts[i] = t

        # B2: recommended images hint (small grey text below name)
        hint = LANDMARK_VISIBILITY.get(name, '')
        vt = ax_list.text(0.09, y - row_h*0.22,
                          f'img {hint}' if hint else '',
                          va='center', color='#666688', fontsize=6,
                          transform=ax_list.transAxes)
        vis_texts[i] = vt

    def _refresh_list():
        for i, name in enumerate(names):
            is_locked = name in locked
            if is_locked:
                u, v = picked[name]
                sym = '🔒'; col = '#888888'; bg_col = '#222233'
                hint_col = '#444455'
                extra = f'  ({u:.0f}, {v:.0f})'
            elif name in picked:
                u, v = picked[name]
                sym = '✓'; col = '#55ff88'; bg_col = '#1a3a1a'
                hint_col = '#336633'
                extra = f'  ({u:.0f}, {v:.0f})'
            elif name in skipped:
                sym = '–'; col = '#666666'; bg_col = '#1e1e1e'
                hint_col = '#444444'
                extra = '  [skip]'
            elif i == sel_idx:
                sym = '▶'; col = '#ffff55'; bg_col = '#3a3a1a'
                hint_col = '#888855'
                extra = ''
            else:
                sym = '○'; col = '#aaaacc'; bg_col = '#2a2a4a'
                hint_col = '#555577'
                extra = ''
            list_texts[i].set_text(f'{sym}  {name}{extra}')
            list_texts[i].set_color(col)
            list_bg[i].set_facecolor(bg_col)
            vis_texts[i].set_color(hint_col)

        n_fix_done = sum(1 for nm in fix_names if nm in picked)
        n_fix_total = len(fix_names) if fix_names else 0
        if fix_names:
            ax_list.set_title(
                f'Fix mode: {n_fix_done}/{n_fix_total} re-picked  '
                f'({len(locked)} locked)',
                color='#ffcc44', fontsize=8, pad=2)
        else:
            done = len(picked) + len(skipped)
            ax_list.set_title(
                f'{done}/{n_lm}  ({len(picked)} picked  {len(skipped)} skipped)',
                color='#cccccc', fontsize=8, pad=2)
        fig.canvas.draw_idle()

    def _refresh_img():
        for a in list(scat_artists.values()) + list(label_artists.values()):
            a.remove()
        scat_artists.clear(); label_artists.clear()
        for name, (u, v) in picked.items():
            if name in locked:
                scat_artists[name]  = ax_img.plot(u, v, '+', color='#888888',
                                                   ms=12, mew=1.5)[0]
                label_artists[name] = ax_img.text(
                    u+10, v-10, name, color='#888888', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.4),
                    visible=show_labels[0])
            else:
                scat_artists[name]  = ax_img.plot(u, v, 'g+', ms=14, mew=2)[0]
                label_artists[name] = ax_img.text(
                    u+10, v-10, name, color='lime', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6),
                    visible=show_labels[0])
        cur_name = names[sel_idx]
        hint = LANDMARK_VISIBILITY.get(cur_name, '')
        vis_note = f'  [img {hint}]' if hint else ''
        if cur_name not in picked and cur_name not in skipped:
            ax_img.set_title(
                title_base + f'\n→ Pick: {cur_name}{vis_note}',
                color='yellow', fontsize=8, pad=4)
        else:
            ax_img.set_title(title_base, color='white', fontsize=8, pad=4)
        fig.canvas.draw_idle()

    def _update_loupe(u, v):
        u_i, v_i = int(round(u)), int(round(v))
        y0 = max(v_i - LOUPE_HALF, 0);  y1 = min(v_i + LOUPE_HALF, img_h)
        x0 = max(u_i - LOUPE_HALF, 0);  x1 = min(u_i + LOUPE_HALF, img_w)
        patch = img[y0:y1, x0:x1]
        pad_t = max(LOUPE_HALF - v_i, 0); pad_b = max(v_i + LOUPE_HALF - img_h, 0)
        pad_l = max(LOUPE_HALF - u_i, 0); pad_r = max(u_i + LOUPE_HALF - img_w, 0)
        if any([pad_t, pad_b, pad_l, pad_r]):
            patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                           mode='constant', constant_values=30)
        loupe_im.set_data(patch)
        loupe_im.set_extent([-0.5, patch.shape[1]-0.5, patch.shape[0]-0.5, -0.5])

    def on_move(event):
        if event.inaxes != ax_img or event.xdata is None:
            return
        u, v = event.xdata, event.ydata
        # A1: live coords in right panel (not title — keeps title clean)
        coord_txt.set_text(f'u = {u:.1f}   v = {v:.1f}')
        # A4: crosshair
        ch_h.set_ydata([v, v]);  ch_v.set_xdata([u, u])
        # A2: loupe
        _update_loupe(u, v)
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal sel_idx
        if event.inaxes != ax_img or event.button != 1:
            return
        cur_name = names[sel_idx]
        if cur_name in locked:
            print(f'  {cur_name} is locked — use ↑↓ to move to a fix landmark')
            return
        u, v = event.xdata, event.ydata
        picked[cur_name] = (u, v)
        skipped.discard(cur_name)
        print(f'  [{sel_idx+1}/{n_lm}] {cur_name}: u={u:.1f}  v={v:.1f}')
        # advance to next unpicked, non-locked, non-skipped fix landmark (if in fix mode)
        # otherwise any unpicked/unskipped landmark
        def _is_done(nm):
            if fix_names:
                return nm not in fix_names or nm in picked or nm in skipped
            return nm in picked or nm in skipped
        for step in range(1, n_lm + 1):
            nxt = (sel_idx + step) % n_lm
            if not _is_done(names[nxt]) and names[nxt] not in locked:
                sel_idx = nxt; break
        _refresh_list(); _refresh_img()

    def on_key(event):
        nonlocal sel_idx
        if event.key in ('enter', 'q'):
            pts_out   = [(u, v) for name in names if name in picked for u, v in [picked[name]]]
            names_out = [name   for name in names if name in picked]
            _save(pts_out, out_path, names_out)
            plt.close()
        elif event.key == 'up':
            sel_idx = (sel_idx - 1) % n_lm
            _refresh_list(); _refresh_img()
        elif event.key == 'down':
            sel_idx = (sel_idx + 1) % n_lm
            _refresh_list(); _refresh_img()
        elif event.key == ' ':
            cur_name = names[sel_idx]
            if cur_name in locked:
                print(f'  {cur_name} is locked — cannot skip')
                return
            skipped.add(cur_name); picked.pop(cur_name, None)
            print(f'  Skipped: {cur_name}')
            sel_idx = (sel_idx + 1) % n_lm
            _refresh_list(); _refresh_img()
        elif event.key == 'z':
            # Only undo non-locked picks
            non_locked_picks = [nm for nm in picked if nm not in locked]
            if non_locked_picks:
                last = non_locked_picks[-1]
                del picked[last]
                skipped.discard(last)
                if last in names:
                    sel_idx = names.index(last)
                print(f'  Undo: removed {last}  →  back to [{sel_idx+1}] {last}')
                _refresh_list(); _refresh_img()
        elif event.key == 'l':
            show_labels[0] = not show_labels[0]
            for a in label_artists.values():
                a.set_visible(show_labels[0])
            state = 'ON' if show_labels[0] else 'OFF'
            print(f'  Labels {state}')
            fig.canvas.draw_idle()

    _refresh_list(); _refresh_img()
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event',  on_click)
    fig.canvas.mpl_connect('key_press_event',      on_key)
    plt.show()


def _save(points, path, names=None):
    if not points:
        print('No points — not saved.')
        return
    arr = np.array(points)
    hdr = ' '.join(names) if names else 'u v'
    np.savetxt(path, arr, fmt='%.4f', header=hdr)
    print(f'\nSaved {len(points)} points → {path}')


if __name__ == '__main__':
    main()

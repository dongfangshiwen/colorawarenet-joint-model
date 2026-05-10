import sys
import os
import argparse
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QRectF, QPointF, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QAction, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QListWidget, QCheckBox, QSpinBox, QMessageBox, QMainWindow,
    QToolBar, QStatusBar, QSlider, QSplitter
)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pil_to_qpixmap(pil_img: Image.Image):
    data = pil_img.convert('RGBA').tobytes('raw', 'RGBA')
    qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


class ImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img = None               # 原始 BGR ndarray
        self.qpix = None              # QPixmap for display
        self.scale = 1.0
        self.offset = QPointF(0, 0)   # panning offset in widget coords
        self.dragging = False
        self.last_pan = None

        # polygons in image coordinates: list of list of (x,y) ints
        self.polygons = []
        # current polygon in image coords
        self.current_pts = []

        # appearance
        self.point_radius = 6  # screen-ish pixels (approx), adjusted in paintEvent when drawing in image space
        self.poly_pen = QPen(QColor(0, 200, 0), 2)
        self.curr_pen = QPen(QColor(200, 30, 30), 2)
        self.fill_color = QColor(0, 200, 0, 80)

        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

    def load_image(self, img_path: Path):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        # convert PNG with alpha to RGB (drop alpha) to avoid libpng warnings
        if bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        self.img = bgr
        h, w = bgr.shape[:2]
        # create QPixmap
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self.qpix = QPixmap.fromImage(qimg)
        # reset view to fit widget
        self.scale = min(self.width() / max(1.0, w), self.height() / max(1.0, h), 1.0)
        self.offset = QPointF((self.width() - w * self.scale) / 2.0, (self.height() - h * self.scale) / 2.0)
        self.polygons = []
        self.current_pts = []
        self.update()

    def sizeHint(self):
        return QSize(1000, 700)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.qpix is None:
            painter.end()
            return
        # draw image scaled and translated
        painter.save()
        try:
            painter.translate(self.offset)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(0, 0, self.qpix)

            # draw polygons (filled) in image coordinate space
            painter.setPen(self.poly_pen)
            painter.setBrush(self.fill_color)
            for poly in self.polygons:
                if len(poly) >= 3:
                    qpts = [QPointF(x, y) for (x, y) in poly]
                    polygon = QPolygonF(qpts)
                    painter.drawPolygon(polygon)

            # draw current polygon (red lines/points) also in image coordinate space
            painter.setPen(self.curr_pen)
            painter.setBrush(Qt.NoBrush)
            if len(self.current_pts) > 0:
                prev = None
                for p in self.current_pts:
                    cx = float(p[0])
                    cy = float(p[1])
                    # radius in image space so visual size stays roughly constant on screen
                    r = max(1.0, self.point_radius / max(1e-6, self.scale))
                    painter.drawEllipse(QPointF(cx, cy), r, r)
                    if prev is not None:
                        painter.drawLine(QPointF(prev[0], prev[1]), QPointF(cx, cy))
                    prev = (cx, cy)
        finally:
            # ensure we always restore/end painter to avoid saved-state leaks
            try:
                painter.restore()
            except Exception:
                pass
            try:
                painter.end()
            except Exception:
                pass

    def widget_to_image(self, wx, wy):
        # convert widget coords click->image coords
        ix = (wx - self.offset.x()) / self.scale
        iy = (wy - self.offset.y()) / self.scale
        return int(round(ix)), int(round(iy))

    def image_to_widget(self, ix, iy):
        wx = ix * self.scale + self.offset.x()
        wy = iy * self.scale + self.offset.y()
        return wx, wy

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.position().x(), event.position().y()
            ix, iy = self.widget_to_image(x, y)
            # clamp
            h, w = self.img.shape[:2]
            ix = max(0, min(w - 1, ix)); iy = max(0, min(h - 1, iy))
            self.current_pts.append((ix, iy))
            self.update()
        elif event.button() == Qt.RightButton:
            # start panning
            self.dragging = True
            self.last_pan = event.position()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.dragging = False
            self.last_pan = None

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_pan is not None:
            cur = event.position()
            dx = cur.x() - self.last_pan.x()
            dy = cur.y() - self.last_pan.y()
            self.offset += QPointF(dx, dy)
            self.last_pan = cur
            self.update()

    def wheelEvent(self, event):
        # zoom about mouse position
        delta = event.angleDelta().y()
        if delta == 0 or self.qpix is None:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        old_scale = self.scale
        new_scale = max(0.05, min(10.0, self.scale * factor))
        # adjust offset so mouse pos remains on same image pixel
        mouse_pos = event.position()
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
        img_x_before = (mouse_x - self.offset.x()) / old_scale
        img_y_before = (mouse_y - self.offset.y()) / old_scale
        self.scale = new_scale
        self.offset = QPointF(mouse_x - img_x_before * new_scale, mouse_y - img_y_before * new_scale)
        self.update()

    def close_current_polygon(self):
        if len(self.current_pts) >= 3:
            # current_pts are already image coords; append as polygon
            self.polygons.append(list(self.current_pts))
            self.current_pts = []
            self.update()

    def undo_point(self):
        if len(self.current_pts) > 0:
            self.current_pts.pop()
        else:
            if len(self.polygons) > 0:
                self.polygons.pop()
        self.update()

    def clear_polygons(self):
        self.current_pts = []
        self.polygons = []
        self.update()

    # def build_mask(self):
    #     if self.img is None:
    #         return None
    #     h, w = self.img.shape[:2]
    #     mask = np.zeros((h, w), dtype=np.uint8)
    #     for poly in self.polygons:
    #         pts = np.array(poly, dtype=np.int32)
    #         if pts.shape[0] >= 3:
    #             cv2.fillPoly(mask, [pts], 255)
    #     return mask

    def build_mask(self):
        if self.img is None:
            return None
        h, w = self.img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self.polygons:
            pts = np.array(poly, dtype=np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], 255)
        # 订正：归一化为 float 0~1
        mask = mask.astype(np.float32) / 255.0
        return mask
    
    def grabcut_refine(self, iter_count=5):
        if self.img is None:
            return None
        h, w = self.img.shape[:2]
        if len(self.polygons) == 0 and len(self.current_pts) < 3:
            return self.build_mask()
        # combine polygons and current_pts
        tmp = list(self.polygons)
        if len(self.current_pts) >= 3:
            tmp.append(list(self.current_pts))
        gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        for poly in tmp:
            pts = np.array(poly, dtype=np.int32)
            if pts.shape[0] >= 3:
                m = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m, [pts], 1)
                gc_mask[m == 1] = cv2.GC_PR_FGD
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            img_copy = self.img.copy()
            cv2.grabCut(img_copy, gc_mask, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)
        except Exception as e:
            print('GrabCut failed:', e)
            return self.build_mask()
        res = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        # find contours and convert to polygons
        contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_polys = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            cnt = cnt.squeeze(1)
            pts = [(int(x), int(y)) for x, y in cnt]
            if len(pts) >= 3:
                new_polys.append(pts)
        if len(new_polys) > 0:
            self.polygons = new_polys
            self.current_pts = []
        # else keep previous polygons
        self.update()
        return self.build_mask()


class MainWindow(QMainWindow):
    def __init__(self, images_dir: Path, masks_dir: Path, overwrite=False, skip_existing=True):
        super().__init__()
        self.setWindowTitle('斑马线标注器 (PySide6)')
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        ensure_dir(self.masks_dir)
        self.overwrite = overwrite
        self.skip_existing = skip_existing

        self.image_files = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if len(self.image_files) == 0:
            raise RuntimeError('没有在 images 目录找到图片')
        self.idx = 0

        # UI
        self.canvas = ImageCanvas()

        btn_prev = QPushButton('上一张')
        btn_next = QPushButton('下一张')
        btn_save = QPushButton('保存 mask')
        btn_gc = QPushButton('GrabCut 精修')
        btn_close = QPushButton('闭合多边形')
        btn_undo = QPushButton('撤销点/多边形')
        btn_clear = QPushButton('清空多边形')

        btn_prev.clicked.connect(self.on_prev)
        btn_next.clicked.connect(self.on_next)
        btn_save.clicked.connect(self.on_save)
        btn_gc.clicked.connect(self.on_grabcut)
        btn_close.clicked.connect(self.on_close_poly)
        btn_undo.clicked.connect(self.on_undo)
        btn_clear.clicked.connect(self.on_clear)

        self.skip_checkbox = QCheckBox('跳过已存在 mask')
        self.skip_checkbox.setChecked(self.skip_existing)
        self.skip_checkbox.stateChanged.connect(self.on_skip_toggle)
        self.overwrite_checkbox = QCheckBox('覆盖已存在 mask')
        self.overwrite_checkbox.setChecked(self.overwrite)
        self.overwrite_checkbox.stateChanged.connect(self.on_overwrite_toggle)

        self.gc_iters_spin = QSpinBox(); self.gc_iters_spin.setRange(1, 50); self.gc_iters_spin.setValue(5)

        # polygon list
        self.poly_list = QListWidget()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.canvas)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(btn_prev)
        ctrl_layout.addWidget(btn_next)
        ctrl_layout.addWidget(btn_save)
        ctrl_layout.addWidget(btn_gc)
        ctrl_layout.addWidget(btn_close)
        ctrl_layout.addWidget(btn_undo)
        ctrl_layout.addWidget(btn_clear)

        opt_layout = QHBoxLayout()
        opt_layout.addWidget(self.skip_checkbox)
        opt_layout.addWidget(self.overwrite_checkbox)
        label_gc = QLabel('GrabCut iters:')
        opt_layout.addWidget(label_gc)
        opt_layout.addWidget(self.gc_iters_spin)

        left_layout.addLayout(ctrl_layout)
        left_layout.addLayout(opt_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel('当前图片:'))
        self.lbl_image_name = QLabel('')
        right_layout.addWidget(self.lbl_image_name)
        right_layout.addWidget(QLabel('多边形列表:'))
        right_layout.addWidget(self.poly_list)

        # top-level
        container = QSplitter()
        left_w = QWidget(); left_w.setLayout(left_layout)
        right_w = QWidget(); right_w.setLayout(right_layout)
        container.addWidget(left_w)
        container.addWidget(right_w)
        container.setStretchFactor(0, 4); container.setStretchFactor(1, 1)

        self.setCentralWidget(container)
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # load image AFTER UI widgets are created to avoid attribute errors
        self.load_current_image()
        self.update_status()

    def update_status(self, text: str = None):
        if text is None:
            text = f'Idx {self.idx+1}/{len(self.image_files)}: {self.image_files[self.idx].name}'
        self.status.showMessage(text)

    def load_current_image(self):
        p = self.image_files[self.idx]
        self.canvas.load_image(p)
        if hasattr(self, 'lbl_image_name') and self.lbl_image_name is not None:
            self.lbl_image_name.setText(str(p.name))
        self.update_poly_list()

    def on_prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_current_image()

    def on_next(self):
        # auto-save prompt? we just move to next and warn if unsaved
        if self.idx < len(self.image_files)-1:
            # if skip_existing enabled and next has mask, skip ahead
            self.idx += 1
            if self.skip_checkbox.isChecked():
                while self.idx < len(self.image_files) and (self.masks_dir / (self.image_files[self.idx].stem + '.png')).exists():
                    self.idx += 1
                    if self.idx >= len(self.image_files):
                        break
            if self.idx >= len(self.image_files):
                QMessageBox.information(self, 'Done', '已达到末尾或所有未标注图片都已处理。')
                self.idx = max(0, len(self.image_files)-1)
            self.load_current_image()

    def on_save(self):
        cur = self.image_files[self.idx]
        out = self.masks_dir / (cur.stem + '.png')
        if out.exists() and not self.overwrite_checkbox.isChecked():
            reply = QMessageBox.question(self, '已存在', f'{out.name} 已存在，是否覆盖？', QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        mask = self.canvas.build_mask()
        if mask is None:
            QMessageBox.warning(self, '错误', '当前没有加载图像')
            return
        cv2.imwrite(str(out), mask)
        self.update_status(f'Saved {out.name}')

    def on_grabcut(self):
        iters = self.gc_iters_spin.value()
        self.canvas.grabcut_refine(iter_count=iters)
        self.update_poly_list()
        self.update_status('GrabCut 完成')

    def on_close_poly(self):
        self.canvas.close_current_polygon()
        self.update_poly_list()

    def on_undo(self):
        self.canvas.undo_point()
        self.update_poly_list()

    def on_clear(self):
        self.canvas.clear_polygons()
        self.update_poly_list()

    def on_skip_toggle(self, state):
        self.skip_existing = (state == Qt.Checked)

    def on_overwrite_toggle(self, state):
        self.overwrite = (state == Qt.Checked)

    def update_poly_list(self):
        self.poly_list.clear()
        for i, poly in enumerate(self.canvas.polygons):
            self.poly_list.addItem(f'poly {i+1}: {len(poly)} pts')

    # optional: export current image polygons into simple COCO-like json (polygons only)
    def export_current_coco(self, out_json_path: Path):
        p = self.image_files[self.idx]
        polys = []
        for poly in self.canvas.polygons:
            # flatten points
            flat = [coord for pt in poly for coord in pt]
            polys.append(flat)
        item = {
            'file_name': p.name,
            'width': int(self.canvas.img.shape[1]),
            'height': int(self.canvas.img.shape[0]),
            'polygons': polys
        }
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='PySide6 based interactive mask maker for zebra lines')
    parser.add_argument('--images', '-i', default='datasets/clear', help='Images directory (jpg/png)')
    parser.add_argument('--masks', '-m', default='datasets/masks', help='Output masks directory (png)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing masks')
    parser.add_argument('--skip', action='store_true', help='Skip files that already have masks')
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    if not images_dir.exists():
        print('Images dir not found:', images_dir)
        return
    ensure_dir(masks_dir)

    app = QApplication(sys.argv)
    win = MainWindow(images_dir, masks_dir, overwrite=args.overwrite, skip_existing=args.skip)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

# python interactive_mask_maker.py --images datasets/fog_images --masks datasets/masks_
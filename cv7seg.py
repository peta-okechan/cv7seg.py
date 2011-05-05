#!/usr/bin/env python
# -*- coding: utf-8 -*-

u'''
WebカメラとOpenCVを使って7セグメントLEDの数値を読み取る
'''

import os, time
from ConfigParser import SafeConfigParser
from PIL import Image
import Tkinter as tk
import ImageTk as itk
import cv
try:
    import cPickle as pickle
except ImportError:
    import pickle


class App(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.init()
    
    def init(self):
        # メニューバーの設定
        self._menu = tk.Menu(self)
        self._mode_menu = tk.Menu(self._menu, tearoff = False)
        self._mode_menu.add_command(label = u'正規化', command = lambda:self.showUI(NormalizeUI))
        self._mode_menu.add_command(label = u'学習', command = lambda:self.showUI(TemplateUI))
        self._mode_menu.add_command(label = u'ロギング', command = lambda:self.showUI(LoggingUI))
        self._menu.add_cascade(label = u'モード', menu = self._mode_menu)
        self.master.config(menu = self._menu)
        
        # UIの表示
        self._ui = None
        if config.general.last_ui:
            self.showUI(config.general.last_ui)
        else:
            self.showUI(NormalizeUI)
    
    def showUI(self, uitype):
        if self._ui:
            self._ui.destroy()
        self._ui = uitype()
        self._ui.pack(fill = tk.BOTH)
        config.general.last_ui = uitype


class TimedFrame(tk.Frame):
    u'''一定間隔で繰り返しメソッドを実行する機能を持つFrameクラス'''
    def __init__(self, master = None):
        tk.Frame.__init__(self, master)
        self._timings = []
    
    def addTiming(self, func, interval, *args, **kwargs):
        u'''一定間隔で実行する関数を登録'''
        self._timings.append(A(func = func, interval = interval, next_time = None, args = args, kwargs = kwargs))
    
    def startTimer(self):
        u'''タイマーの開始'''
        self.after(1000 / config.general.fps, self.framein)
            
    def framein(self):
        u'''一定間隔の処理を誤差の蓄積なしに実行する'''
        for t in self._timings:
            if t.next_time is None:
                # 初回の処理をすぐに実行するため現在時間を設定
                t.next_time = time.time()
            if time.time() >= t.next_time:
                # 次の処理を実行する時間なので実行する
                t.next_time += t.interval
                t.func(*t.args, **t.kwargs)
        self.after(1000 / config.general.fps, self.framein)
        

class NormalizeUI(TimedFrame):
    u'''正規化設定用UI'''
    def __init__(self, *args, **kwargs):
        TimedFrame.__init__(self, *args, **kwargs)
        self.pack()
        self.init()
        self.startTimer()
    
    def init(self):
        # カメラの準備
        self._camera = cv.CaptureFromCAM(0)
        
        # カメラ画像表示用Canvasなどの準備
        self._image = tk.PhotoImage(width = config.canvas.width, height = config.canvas.height)
        self._canvas = tk.Canvas(self, width = config.canvas.width, height = config.canvas.height)
        self._canvas.create_image(config.canvas.width / 2, config.canvas.height / 2, image = self._image, tags = 'image')
        self._canvas.pack(expand = 1, fill = tk.BOTH)
        self._canvas.bind('<ButtonRelease-1>', self.mouseUp)
        
        # ボタン
        self._main_button = tk.Button(self, text = u'確定', command = self.fixation)
        self._main_button.pack(side = tk.TOP, fill = tk.BOTH)
        widgetEnable(self._main_button, False)
        self._cancel_button = tk.Button(self, text = u'キャンセル', command = self.cancel)
        self._cancel_button.pack(side = tk.TOP, fill = tk.BOTH)
        
        # クリック座標の情報
        self._clicks = []
        
        # 画像をフィルタするための変数
        self._clip_rect = None
        self._perspective_points = None
        
        # カメラ画像の更新を1秒間隔にする
        self.addTiming(self.showImage, 1)
        
    def showImage(self):
        u'''カメラ画像の表示'''
        captured = cv.QueryFrame(self._camera)
        self._image = CvMat2TkImage(self.filter(captured))
        self._canvas.itemconfigure('image', image = self._image)
    
    def mouseUp(self, e):
        if len(self._clicks) < 4:
            size = 4
            self._clicks.append(A(x = e.x, y = e.y))
            self._canvas.create_oval(
                e.x - size / 2, e.y - size / 2,
                e.x + size / 2, e.y + size / 2,
                fill = 'red',
                tags = 'clicks',
            )
        if len(self._clicks) >= 4:
            self._clip_rect, self._perspective_points = Points2Rect(self._clicks)
            self._canvas.delete('clicks')
            widgetEnable(self._main_button, True)
    
    def fixation(self):
        u'''確定処理'''
        assert len(self._clicks) == 4
        config.normalize.points = self._clicks
    
    def cancel(self):
        u'''キャンセル処理'''
        widgetEnable(self._main_button, False)
        self._clicks = []
        self._clip_rect = None
        self._perspective_points = None
    
    def filter(self, cvmat):
        u'''画像をフィルタする'''
        # サイズ調整
        thumbnail = cv.CreateMat(config.canvas.height, config.canvas.width, cv.CV_8UC3)
        cv.Resize(cvmat, thumbnail)
        
        if self._clip_rect and self._perspective_points:
            return NormalizeImage(thumbnail, self._clip_rect, self._perspective_points)
        else:
            return thumbnail


class TemplateUI(TimedFrame):
    u'''テンプレート設定用UI'''
    def __init__(self, *args, **kwargs):
        TimedFrame.__init__(self, *args, **kwargs)
        self.pack()
        self.init()
        self.startTimer()
    
    def init(self):
        if not config.normalize.points or len(config.normalize.points) < 4:
            self._label = tk.Label(self, text = u'まだ正規化が済んでいません。\n正規化を行ってください。')
            self._label.pack()
            return
        
        if not config.template.images:
            config.template.images = [None for i in xrange(10)]
        
        # カメラの準備
        self._camera = cv.CaptureFromCAM(0)
        
        # カメラ画像表示用Canvasなどの準備
        self._cvmat = None
        self._image = tk.PhotoImage(width = config.canvas.width, height = config.canvas.height)
        self._canvas = tk.Canvas(self, width = config.canvas.width, height = config.canvas.height)
        self._canvas.create_image(config.canvas.width / 2, config.canvas.height / 2, image = self._image, tags = 'image')
        self._canvas.pack(expand = 1, fill = tk.BOTH)
        self._canvas.tag_bind('image', '<ButtonPress-1>', self.mouseDown)
        self._canvas.tag_bind('image', '<B1-Motion>', self.mouseDrag)
        self._canvas.tag_bind('image', '<ButtonRelease-1>', self.mouseUp)
        
        # ボタン
        self._buttons = []
        for i in xrange(10):
            command = (lambda id: lambda: self.fixation(id))(i)
            button = tk.Button(self, text = u'%d' % i, command = command)
            button.pack(side = tk.LEFT)
            self._buttons.append(button)
            # ボタン画像をセーブデータから復元する
            cvimageinfo = config.template.images[i]
            if cvimageinfo:
                cvmat = cv.CreateMatHeader(cvimageinfo.rows, cvimageinfo.cols, cvimageinfo.type)
                cv.SetData(cvmat, cvimageinfo.data)
                self.setButtonImage(i, cvmat)
        self.allButtonEnable(False)
        
        # マウス座標の情報
        self._mouse_down = None
        self._mouse_up = None
        
        # 画像をフィルタするための変数
        self._clip_rect, self._perspective_points = Points2Rect(config.normalize.points)
        
        # カメラ画像の更新を1秒間隔にする
        self.addTiming(self.showImage, 1)
        
    def showImage(self):
        u'''カメラ画像の表示'''
        captured = cv.QueryFrame(self._camera)
        self._cvmat = self.filter(captured)
        self._image = CvMat2TkImage(self._cvmat)
        self._canvas.itemconfigure('image', image = self._image)
    
    def mouseDown(self, e):
        self._mouse_down = A(x = e.x, y = e.y)
        self._mouse_up = None
        self._canvas.delete('rect')
        self.allButtonEnable(False)
    
    def mouseDrag(self, e):
        if self._mouse_down:
            self._canvas.delete('rect')
            self._canvas.create_rectangle(
                self._mouse_down.x, self._mouse_down.y,
                e.x, e.y,
                tags = 'rect',
            )
    
    def mouseUp(self, e):
        if self._mouse_down and self._mouse_down.x != e.x and self._mouse_down.y != e.y:
            self._mouse_up = A(x = e.x, y = e.y)
            self._canvas.delete('rect')
            self._canvas.create_rectangle(
                self._mouse_down.x, self._mouse_down.y,
                e.x, e.y, fill= 'green', stipple = 'gray25',
                tags = 'rect',
            )
            self.allButtonEnable(True)

    def fixation(self, id):
        u'''確定処理'''
        assert self._mouse_down and self._mouse_up
        if self._cvmat:
            p1 = self.canvasCoord2ImageCoord(self._mouse_down)
            p2 = self.canvasCoord2ImageCoord(self._mouse_up)
            clipped = ClipImage(self._cvmat, p1, p2)
            config.template.images[id] = A(
                rows = clipped.rows, cols = clipped.cols,
                type = clipped.type, data = clipped.tostring(),
            )
            self.setButtonImage(id, clipped)
    
    def filter(self, cvmat):
        u'''画像をフィルタする'''
        # サイズ調整
        thumbnail = cv.CreateMat(config.canvas.height, config.canvas.width, cv.CV_8UC3)
        cv.Resize(cvmat, thumbnail)
        return NormalizeImage(thumbnail, self._clip_rect, self._perspective_points)
    
    def allButtonEnable(self, enable):
        u'''全てのボタンをオンオフする'''
        for button in self._buttons:
            widgetEnable(button, enable)
    
    def setButtonImage(self, id, cvmat):
        u'''ボタンに背景画像を設定する'''
        self._buttons[id].clipped = CvMat2TkImage(cvmat)
        self._buttons[id].configure(image = self._buttons[id].clipped)
    
    def canvasCoord2ImageCoord(self, canvas_coord):
        u'''キャンバス座標を画像の座標に変換'''
        return A(
            x = canvas_coord.x - (config.canvas.width - self._clip_rect.width) / 2,
            y = canvas_coord.y - (config.canvas.height - self._clip_rect.height) / 2,
        )


class LoggingUI(TimedFrame):
    u'''ロギング用UI'''
    def __init__(self, *args, **kwargs):
        TimedFrame.__init__(self, *args, **kwargs)
        self.pack()
        self.init()
        self.startTimer()
    
    def init(self):
        if not config.normalize.points:
            self._label = tk.Label(self, text = u'まだ正規化が済んでいません。\n正規化を行ってください。')
            self._label.pack()
            return

        if not config.template.images or len(config.template.images) < 10 or not all(config.template.images):
            self._label = tk.Label(self, text = u'まだ学習が済んでいません。\n学習を行ってください。')
            self._label.pack()
            return
        
        # テンプレートの読み込み
        self.loadTemplates()
        
        # カメラの準備
        self._camera = cv.CaptureFromCAM(0)
        
        # 左側UI
        frame1 = tk.Frame(self)
        frame1.pack(side = tk.LEFT, expand = 1, fill = tk.BOTH)
        
        # カメラ画像表示用Canvasなどの準備
        self._cvmat = None
        self._image = tk.PhotoImage(width = config.canvas.width, height = config.canvas.height)
        self._canvas = tk.Canvas(frame1, width = config.canvas.width, height = config.canvas.height)
        self._canvas.create_image(config.canvas.width / 2, config.canvas.height / 2, image = self._image, tags = 'image')
        self._canvas.pack(expand = 1, fill = tk.BOTH)
        
        # ボタン
        self._main_button = tk.Button(frame1)
        self._main_button.pack(side = tk.TOP, fill = tk.BOTH)
        self.logStop()
        
        # 右側UI
        frame2 = tk.Frame(self)
        frame2.pack(side = tk.RIGHT, expand = 1, fill = tk.BOTH)
        
        # ログデータ表示領域
        self._graph = tk.Canvas(frame2, width = config.canvas.width, height = config.canvas.height, bg = 'white')
        self._graph.pack(expand = 1, fill = tk.BOTH)
        self._bar_graph = BarGraph(A(width = config.canvas.width, height = config.canvas.height), config.logging.graph_max_count)
        
        # ボタン
        self._out_button = tk.Button(frame2, text = u'生データ')
        self._out_button.pack(side = tk.LEFT)
        
        # 画像をフィルタするための変数
        self._clip_rect, self._perspective_points = Points2Rect(config.normalize.points)
        
        # ロギング開始、停止のフラグ
        self._take_log = False
        
        # カメラ画像の更新を1秒間隔にする
        self.addTiming(self.showImage, 1)
    
    def setValue(self, value):
        if value is None: value = 0
        self._bar_graph.setValue(value)
        
        self._graph.delete('bar')
        for bar in self._bar_graph.getAllBars():
            self._graph.create_rectangle(
                bar.p1.x, bar.p1.y,
                bar.p2.x, bar.p2.y,
                fill = 'green', stipple = 'gray25',
                tags = 'bar'
            )
    
    def loadTemplates(self):
        u'''テンプレート画像の読み込み'''
        self._templates = []
        for i, cvimageinfo in enumerate(config.template.images):
            cvmat = cv.CreateMatHeader(cvimageinfo.rows, cvimageinfo.cols, cvimageinfo.type)
            cv.SetData(cvmat, cvimageinfo.data)
            self._templates.append(A(
                image = cv.GetImage(cvmat),
                number = i,
                result = None,
            ))
    
    def showImage(self):
        u'''カメラ画像の表示'''
        captured = cv.QueryFrame(self._camera)
        self._cvmat = self.filter(captured)
        self._image = CvMat2TkImage(self._cvmat)
        self._canvas.itemconfigure('image', image = self._image)
        if self._take_log:
            self.logging()
    
    def logging(self):
        u'''ログを取る'''
        target = self._cvmat
        digits_sieve = DigitsSieve()
        for template in self._templates:
            if not template.result:
                # マッチング結果保存用領域の準備
                template.result = cv.CreateImage(
                    (target.width - template.image.width + 1, target.height - template.image.height + 1),
                    cv.IPL_DEPTH_32F, 1,
                )
            
            cv.MatchTemplate(target, template.image, template.result, config.logging.match_method)

            # 数値の読み取り
            minVal, maxVal, minLoc, maxLoc = cv.MinMaxLoc(template.result)
            while maxVal > config.logging.match_threshold:
                # 検出された数値情報の保持
                digits_sieve.push(A(
                    number = template.number,
                    x = maxLoc[0],
                    y = maxLoc[1],
                    width = template.image.width,
                    height = template.image.height,
                    score = maxVal,
                ))
                
                # 現在の位置周辺のスコアをクリアし、次にスコアの高い位置を取得する
                SetReal2DAround(template.result, maxLoc, config.logging.match_exclusion_size, 0.0)
                minVal, maxVal, minLoc, maxLoc = cv.MinMaxLoc(template.result)
        
        self.setValue(digits_sieve.getValue())
        #self._textarea.insert('end', '%d\n' % digits_sieve.getValue())
        #self.log(digits_sieve.getValue(), waiter.getPassage())

    def logStart(self):
        u'''ロギングを開始する'''
        self._main_button.configure(text = u'ストップ', command = self.logStop)
        self._take_log = True
        
    def logStop(self):
        u'''ロギングを停止する'''
        self._main_button.configure(text = u'スタート', command = self.logStart)
        self._take_log = False
        
    def filter(self, cvmat):
        u'''画像をフィルタする'''
        # サイズ調整
        thumbnail = cv.CreateMat(config.canvas.height, config.canvas.width, cv.CV_8UC3)
        cv.Resize(cvmat, thumbnail)
        return NormalizeImage(thumbnail, self._clip_rect, self._perspective_points)
        

def widgetEnable(widget, enable):
    u'''widgetの有効無効を切り替える'''
    if enable:
        widget.configure(state = tk.NORMAL)
    else:
        widget.configure(state = tk.DISABLED) 


def CvMat2TkImage(cvmat):
    u'''CvMatをTk用に変換'''
    return IPLImage2TkImage(cv.GetImage(cvmat))
    

def IPLImage2TkImage(iplimage):
    u'''IPLImageをTk用に変換'''
    assert iplimage.depth == 8 and iplimage.nChannels in [1, 3]
    if iplimage.nChannels == 3:
        image = Image.fromstring('RGB', cv.GetSize(iplimage), iplimage.tostring(), 'raw', 'BGR', iplimage.width * 3, 0)
    elif iplimage.nChannels == 1:
        image = Image.fromstring('L', cv.GetSize(iplimage), iplimage.tostring())
    return itk.PhotoImage(image)


def SetReal2DAround(source_image, loc, size, value):
    u'''locで指定された画素を中心に縦横sizeの範囲にある画素をvalueにする'''
    assert size >= 1 and size % 2 == 1, u'sizeは1以上の奇数である必要があります'    
    rows = xrange(loc[1] - (size - 1) / 2, loc[1] + (size - 1) / 2 + 1)
    cols = xrange(loc[0] - (size - 1) / 2, loc[0] + (size - 1) / 2 + 1)
    for row in rows:
        if 0 <= row < source_image.height:
            for col in cols:
                if 0 <= col < source_image.width:
                    cv.SetReal2D(source_image, row, col, value)


def NormalizeImage(cvmat, cilp_rect, perspective_points):
    u'''読み取りやすくするために画像を正規化する'''
    # 液晶部分の抽出
    lcd = cv.CreateMat(cilp_rect.height, cilp_rect.width, cv.CV_8UC3)
    cv.GetRectSubPix(cvmat, lcd, (cilp_rect.cx, cilp_rect.cy))
    
    # グレイスケール化
    grayed = cv.CreateMat(lcd.height, lcd.width, cv.CV_8UC1)
    cv.CvtColor(lcd, grayed, cv.CV_BGR2GRAY)
    
    # 適応的2値化
    filterd = cv.CreateMat(grayed.height, grayed.width, cv.CV_8UC1)
    cv.AdaptiveThreshold(
        grayed, filterd, 255,
        adaptive_method = cv.CV_ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType = cv.CV_THRESH_BINARY,
        blockSize = 15,
    )
    
    # ゆがみ補正
    transformed = cv.CreateMat(grayed.height, grayed.width, filterd.type)
    matrix = cv.CreateMat(3, 3, cv.CV_32F)
    cv.GetPerspectiveTransform(
        (
            (perspective_points.tl.x, perspective_points.tl.y),
            (perspective_points.tr.x, perspective_points.tr.y),
            (perspective_points.bl.x, perspective_points.bl.y),
            (perspective_points.br.x, perspective_points.br.y)
        ),
        ((0, 0), (filterd.width, 0), (0, filterd.height), (filterd.width, filterd.height)),
        matrix
    )
    cv.WarpPerspective(filterd, transformed, matrix,
        flags = cv.CV_WARP_FILL_OUTLIERS,
        fillval = 255,
    )
    
    return transformed


def ClipImage(cvmat, p1, p2):
    u'''画像を矩形領域で切り抜き'''
    assert p1.x > 0 and p1.y > 0 and p2.x > 0 and p2.y > 0
    width = abs(p2.x - p1.x)
    height = abs(p2.y - p1.y)
    if width > 0 and height > 0:
        x = p1.x + (p2.x - p1.x) / 2
        y = p1.y + (p2.y - p1.y) / 2
        clipped = cv.CreateMat(height, width, cvmat.type)
        cv.GetRectSubPix(cvmat, clipped, (x, y))
        return clipped
    return None
    

def Points2Rect(points):
    u'''与えられた点を含む矩形領域と、上下左右の頂点から一番近い点の相対座標を返す'''
    assert len(points) >= 1
    # p1 = 矩形領域の左上のサイズ
    # p2 = 矩形領域の右下のサイズ
    p1 = A(x = points[0].x, y = points[0].y)
    p2 = A(x = points[0].x, y = points[0].y)
    for p in points:
        p1.x = min(p1.x, p.x)
        p1.y = min(p1.y, p.y)
        p2.x = max(p2.x, p.x)
        p2.y = max(p2.y, p.y)
    # 矩形領域のサイズ
    size = A(width = p2.x - p1.x, height = p2.y - p1.y)
    # 矩形領域の中心座標
    pc = A(x = p1.x + size.width / 2, y = p1.y + size.height / 2)
    
    rect = A(
        x1 = p1.x, y1 = p1.y,
        x2 = p2.x, y2 = p2.y,
        cx = pc.x, cy = pc.y,
        width = size.width, height = size.height,
    )
    
    # 左上、右上、左下、右下からそれぞれ一番近い点を割り出す
    # 左上からの相対座標
    pos = A(
        tl = SubPoint(GetNearestPoint(p1, points), p1),
        tr = SubPoint(GetNearestPoint(A(x = p2.x, y = p1.y), points), p1),
        bl = SubPoint(GetNearestPoint(A(x = p1.x, y = p2.y), points), p1),
        br = SubPoint(GetNearestPoint(p2, points), p1),
    )
    return (rect, pos)


def GetNearestPoint(refpoint, points):
    u'''refpointから一番近い点をpointsから探し出す'''
    assert len(points) >= 1
    nearest_dist = None
    nearest_point = None
    for p in points:
        if nearest_dist is not None:
            dist = (p.x - refpoint.x) ** 2 + (p.y - refpoint.y) ** 2
            if nearest_dist > dist:
                nearest_dist = dist
                nearest_point = p
        else:
            nearest_dist = (p.x - refpoint.x) ** 2 + (p.y - refpoint.y) ** 2
            nearest_point = p
    return nearest_point


def SubPoint(p1, p2):
    u'''Pointの引き算'''
    return A(x = p1.x - p2.x, y = p1.y - p2.y)


class DigitsSieve(object):
    u'''数値情報を受け取り、桁ごとに最適なものを選び出すクラス'''
    def __init__(self):
        self._digits = []
    
    def push(self, number_info):
        for i, digit in enumerate(self._digits):
            succ_w = min(digit.x + digit.width, number_info.x + number_info.width) - max(digit.x, number_info.x)
            whole_w = max(digit.x + digit.width, number_info.x + number_info.width) - min(digit.x, number_info.x)
            succ_rate = float(succ_w) / whole_w
            if succ_rate > 0.2:
                if digit.score < number_info.score:
                    self._digits[i] = number_info
                return
        
        self._digits.append(number_info)
    
    def getDigits(self):
        return [d.number for d in sorted(self._digits, key = lambda n: n.x)]
    
    def getValue(self):
        value = 0
        if self._digits:
            for d in self.getDigits():
                value *= 10
                value += d
            return value
        else:
            None


class BarGraph(object):
    u'''簡易的な棒グラフを表すクラス'''
    def __init__(self, outer_size, max_count):
        assert outer_size.width > 0 and outer_size.height > 0
        assert max_count > 0
        self._outer_size = outer_size
        self._max_count = max_count
        self._values = [0 for i in xrange(max_count)]
        self._max_value = 0
        self.init()
        
    def init(self):
        self._area = A(
            x = float(self._outer_size.width) / 10,
            y = float(self._outer_size.height) / 10,
            width = float(self._outer_size.width) / 10 * 8,
            height = float(self._outer_size.height) / 10 * 8,
        )
        self._bar_width = float(self._area.width) / self._max_count
    
    def setValue(self, value):
        self._values.pop(0)
        self._values.append(value)
        self._max_value = max(self._values)
    
    def getHeight(self, value):
        if self._max_value > 0:
            return float(self._area.height) / self._max_value * value
        else:
            return 0
    
    def getAllBars(self):
        x1 = 0
        for v in self._values:
            yield A(
                p1 = self.yRevAndAbs(A(x = x1, y = self.getHeight(v))),
                p2 = self.yRevAndAbs(A(x = x1 + self._bar_width, y = 0)),
            )
            x1 += self._bar_width
    
    def yRevAndAbs(self, p):
        u'''y軸を反転させて、絶対座標に変換'''
        p.y = self._area.height - p.y
        p.x += self._area.x
        p.y += self._area.y
        return p


class AttrDict(dict):
	"""A dictionary with attribute-style access. It maps attribute access to
	the real dictionary.  """
	def __init__(self, **kwargs):
		dict.__init__(self, kwargs)

	def __getstate__(self):
		return self.__dict__.items()

	def __setstate__(self, items):
		for key, val in items:
			self.__dict__[key] = val

	def __repr__(self):
		return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

	def __setitem__(self, key, value):
		return super(AttrDict, self).__setitem__(key, value)

	def __getitem__(self, name):
		return super(AttrDict, self).__getitem__(name)

	def __delitem__(self, name):
		return super(AttrDict, self).__delitem__(name)

	__getattr__ = __getitem__
	__setattr__ = __setitem__

	def copy(self):
		ch = AttrDict(self)
		return ch
	
	def fullupdate(self, d):
	    stack = [(self, d)]
	    while stack:
	        cur_dst, cur_src = stack.pop()
	        for key in cur_src:
	            if key not in cur_dst:
	                cur_dst[key] = cur_src[key]
	            else:
	                if isinstance(cur_src[key], dict) and isinstance(cur_dst[key], dict):
	                    stack.append((cur_dst[key], cur_src[key]))
	                else:
	                    cur_dst[key] = cur_src[key]
A = AttrDict


# デフォルトの設定値
config = A(
    general = A(
        last_ui = None,
        fps = 30,
    ),
    window = A(
        x = 600, y = 200,
    ),
    canvas = A(
        width = 320, height = 240,
    ),
    normalize = A(
        timing = 1,
        points = [],
        adaptivethreshold_blocksize = 15,
    ),
    template = A(
        images = [],
    ),
    logging = A(
        match_threshold = 0.7,
        match_exclusion_size = 9,
        match_method = cv.CV_TM_CCOEFF_NORMED,
        graph_max_count = 30,
    ),
)
config_file = os.path.splitext(os.path.abspath(__file__))[0] + '.pickle'
try:
    with open(config_file, 'rb') as cf:
        config.fullupdate(pickle.load(cf))
except (IOError, EOFError):
    pass


if __name__ == "__main__":
    try:
        app = App()
        app.master.geometry('+%(x)d+%(y)d' % config.window)
        app.mainloop()
    
    finally:
        with open(config_file, 'wb') as cf:
            pickle.dump(config, cf)
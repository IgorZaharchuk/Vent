#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.0.2
"""
import json
import time
import math
import os
import threading
import requests
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import numpy as np
from collections import deque

# ==================== КОНСТАНТЫ ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(SCRIPT_DIR, 'vent_knowledge.json')
STATE_PATH = os.path.join(SCRIPT_DIR, 'vent_state.json')
DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, 'vent_debug.log')

# Параметры системы (только внутренние настройки алгоритма)
PARAMS = {
    'k_heat_init': 0.000107,           # Начальный коэффициент теплопередачи (1/сек)
    'max_bed': 120,                     # Аппаратное ограничение стола (°C)
    'adapt_start': 60.0,                 # Через сколько секунд начинать адаптацию
    'error5_margin': 3.0,                 # Запас для ошибки 5 (минуты)
    'cooldown_sec': 45.0,                 # Сколько секунд остывания для принудительной адаптации
    'adapt_step': 15,                     # Шаг принудительной адаптации (°C)
    'bed_stable': 3.0,                    # Допуск стабильности стола (°C)
    'max_sessions': 30,                   # Максимум сессий в базе знаний
    'history_points': 100,                 # Точек истории на сессию
    'rate_history_size': 5,                # Размер истории скорости для определения остывания
    'target_hysteresis': 0.5,              # Гистерезис для определения достижения цели (°C)
    'stabilization_delay': 120.0,          # Задержка после стабилизации стола (120 секунд)
}

# ==================== ЛОГГЕР ====================
class Logger:
    def __init__(self, max_bytes=524288):
        self.logger = logging.getLogger('vent_calc')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        formatter.converter = lambda *args: time.gmtime(time.time() + 3*3600)
        
        handler = RotatingFileHandler(
            DEBUG_LOG_PATH, mode='a', maxBytes=max_bytes,
            backupCount=0, encoding='utf-8'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def startup(self, k_heat, samples, bed_rate):
        self.logger.info(f"vent_calc v2.0.2 | k_heat={k_heat:.6f} 1/сек | семпл={samples} | скорость стола: {bed_rate:.1f}°C/мин")

    def status(self, sessions, samples, bed_rate):
        self.logger.info(f"📚 {sessions} сес | семпл={samples} | скорость стола: {bed_rate:.1f}°C/мин")

    def init(self, bed, k_heat, start_temp, start_type, pred_time, bed_rate, heat_time):
        self.logger.info(f"🔄 init bed={bed:.0f}°C | k_heat={k_heat:.6f} | старт={start_temp:.1f}°C({start_type}) | прогноз={pred_time:.0f}мин | скорость стола: {bed_rate:.1f}°C/мин | разогрев: {heat_time:.1f}мин")

    def predict_full(self, pred, elapsed_sec, time_left, target_time, method, is_initial_target):
        target_type = "нач" if is_initial_target else "тек"
        self.logger.info(
            f"📊 t={elapsed_sec:4.0f}с | left={int(time_left)} | цель={target_time:.0f}мин({target_type}) | "
            f"{pred:3}мин ({method})"
        )

    def learned(self, param, old, new, source):
        self.logger.info(f"📈 {param}: {old:.6f}→{new:.6f} ({source})")

    def adapt_up(self, old, new, pred, target_time, stable_since, is_initial_target):
        target_type = "нач" if is_initial_target else "тек"
        self.logger.info(f"⬆️ {old:.0f}→{new:.0f}°C | нужно {pred:.0f}мин, цель {target_time:.0f}мин({target_type}) | стабилен {stable_since:.0f}с")

    def adapt_down(self, old, new, pred, target_time, stable_since, is_initial_target):
        target_type = "нач" if is_initial_target else "тек"
        self.logger.info(f"⬇️ {old:.0f}→{new:.0f}°C | нужно {pred:.0f}мин, цель {target_time:.0f}мин({target_type}) | стабилен {stable_since:.0f}с")

    def adapt_force(self, old, new, reason):
        self.logger.info(f"🔥 {old:.0f}→{new:.0f}°C | {reason}")

    def result(self, temp, rate, bed_set, bed_real, stable):
        rate_str = f"{rate:.2f}" if abs(rate) > 0.01 else "0.00"
        stable_str = "стаб" if stable else "нагр"
        self.logger.info(
            f"✅ {temp:5.1f}°C | {rate_str:5}°C/мин | "
            f"ст={bed_set:3.0f}/{bed_real:3.0f}°C | {stable_str:4}"
        )

    def target_reached(self, current, target, sid):
        self.logger.info(f"🎯 ЦЕЛЬ ДОСТИГНУТА! {current:.1f}°C >= {target:.1f}°C | сес #{sid}")

    def session_closed(self, sid, reason):
        self.logger.info(f"🔒 Сессия #{sid} закрыта: {reason}")

    def ignored_after_target(self, sid):
        self.logger.info(f"⏭️ Вызов проигнорирован - цель уже достигнута (сес #{sid})")

    def external_target_received(self, sid):
        self.logger.info(f"📨 Внешний флаг target_reached=1 получен для сес #{sid}")

    def waiting_for_cooldown(self, sid):
        self.logger.info(f"⏳ Ожидание остывания камеры (сес #{sid})")

    def cooldown_ready(self):
        self.logger.info(f"🔄 Камера остыла, начинаем новую сессию")

    def error_5_set(self, pred, time_left):
        self.logger.warning(f"⚠️ e5 УСТАНОВЛЕНА: нужно {pred:.0f}мин, ост {time_left:.0f}мин")

    def error_5_cleared(self, pred, time_left):
        self.logger.info(f"✅ e5 сброшена: {pred:.0f}мин ≤ {time_left:.0f}мин")

    def stabilization_start(self, sid):
        self.logger.info(f"⏳ Стол стабилизировался, начало отсчета {PARAMS['stabilization_delay']:.0f}с")

    def stabilization_ready(self, sid, stable_since):
        self.logger.info(f"✅ Тепловой поток стабилизирован ({stable_since:.0f}с), можно адаптироваться")

    def session_analysis(self, sid, points, found_samples, bed_rate):
        self.logger.info(f"📊 Сессия #{sid}: {points} точек, {found_samples} семплов k_heat, скорость стола: {bed_rate:.1f}°C/мин")

    def k_heat_update(self, old, new, samples_count):
        self.logger.info(f"📈 k_heat обновлен: {old:.6f} → {new:.6f} (на основе {samples_count} семплов)")

    def bed_rate_update(self, old, new, samples_count):
        self.logger.info(f"📈 Скорость стола обновлена: {old:.1f} → {new:.1f}°C/мин (на основе {samples_count} сессий)")

log = Logger()

# ==================== ФИЛЬТР КАЛМАНА ====================
class KalmanFilter:
    def __init__(self, init_temp, dt=10.0):
        self.x = np.array([init_temp, 0.0])
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.P = np.eye(2) * 100.0
        self.R = 0.1
        self.Q = np.diag([0.001, 0.0001])
        self.rate_ema = 0.0

    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - np.outer(K.flatten(), self.H.flatten())) @ self.P
        
        raw_rate = self.x[1] * 60.0
        self.rate_ema = 0.1 * raw_rate + 0.9 * self.rate_ema
        return self.x[0], self.rate_ema

# ==================== ФИЗИЧЕСКАЯ МОДЕЛЬ ====================
class PhysicsModel:
    """Модель нагрева камеры на основе экспоненциальной асимптоты"""
    def __init__(self, k_heat=PARAMS['k_heat_init']):
        self.k_heat = k_heat

    def time_to_target(self, bed, current, target):
        """Расчет времени до достижения цели (минуты)"""
        if target >= bed:
            return None
        try:
            ratio = (bed - target) / (bed - current)
            if ratio <= 0 or ratio >= 1:
                return None
            t_sec = -math.log(ratio) / self.k_heat
            return t_sec / 60.0
        except Exception:
            return None

    def bed_needed(self, current, target, target_minutes, bed_min):
        """Какая нужна температура стола для достижения цели за заданное время"""
        t_sec = target_minutes * 60
        exp_kt = math.exp(-self.k_heat * t_sec)
        if exp_kt >= 0.99:
            return PARAMS['max_bed']
        
        bed = (target - current * exp_kt) / (1 - exp_kt)
        # Ограничиваем сверху аппаратным максимумом, снизу - переданным bed_min из конфига
        result = max(bed_min, min(PARAMS['max_bed'], bed))
        
        # ОТЛАДКА: логируем расчет
        log.logger.info(f"🔍 bed_needed: current={current:.1f}°C, target={target}°C, time={target_minutes:.1f}мин, k_heat={self.k_heat:.6f}, exp_kt={exp_kt:.4f}, raw_bed={bed:.1f}°C, bed_min={bed_min}, result={result:.1f}°C")
        return result

    def update_k_heat(self, bed, T0, T1, dt):
        """Обновление k_heat по реальным данным (если возможно)"""
        if abs(bed - T0) < 1 or bed <= T1:
            return None
        try:
            ratio = (bed - T1) / (bed - T0)
            if 0.01 < ratio < 0.99:
                new_k = -math.log(ratio) / dt
                return new_k
        except Exception:
            pass
        return None

    def get_k_eq(self):
        """Коэффициент для расчета равновесной температуры"""
        return 0.33 + 1800 * self.k_heat

    def is_reachable(self, amb, target):
        """Проверка, достижима ли цель физически (по равновесной температуре)"""
        k_eq = self.get_k_eq()
        k_eq = max(0.3, min(0.6, k_eq))
        max_eq = amb + k_eq * (PARAMS['max_bed'] - amb)
        return target <= max_eq + 2.0

# ==================== БАЗА ЗНАНИЙ ====================
class KnowledgeBase:
    """База знаний для хранения сессий и калибровки параметров"""
    def __init__(self):
        self.data = self._load()
        self.physics = PhysicsModel(self.get_k_heat())
        self.last_logged_sessions = len(self.data.get('sessions', []))

    def _load(self):
        if os.path.exists(KNOWLEDGE_PATH):
            try:
                with open(KNOWLEDGE_PATH) as f:
                    data = json.load(f)
                if data.get('version') == '2.0' and 'sessions' in data:
                    if 'bed_rates' not in data:
                        data['bed_rates'] = []
                    # Добавляем поле analyzed для сессий, если его нет
                    for session in data['sessions']:
                        if 'analyzed' not in session:
                            session['analyzed'] = False
                    return data
            except Exception as e:
                log.logger.error(f"Ошибка загрузки: {e}")
                backup = KNOWLEDGE_PATH + '.bak.' + str(int(time.time()))
                try:
                    os.rename(KNOWLEDGE_PATH, backup)
                    log.logger.info(f"📦 Создан бэкап: {backup}")
                except:
                    pass
        return self._create_default()

    def _create_default(self):
        return {
            "version": "2.0",
            "k_heat": PARAMS['k_heat_init'],
            "samples": [],
            "bed_rates": [],
            "sessions": []
        }

    def _save(self):
        with lock:
            with open(KNOWLEDGE_PATH + '.tmp', 'w') as f:
                json.dump(self.data, f, indent=2)
            os.replace(KNOWLEDGE_PATH + '.tmp', KNOWLEDGE_PATH)

    def get_k_heat(self):
        return self.data.get('k_heat', PARAMS['k_heat_init'])

    def get_samples_count(self):
        return len(self.data.get('samples', []))

    def get_bed_rate(self):
        rates = self.data.get('bed_rates', [])
        if not rates:
            return 8.0
        return np.median(rates[-20:])

    def _cleanup_sessions(self):
        if len(self.data['sessions']) <= PARAMS['max_sessions']:
            return
        sessions_with_points = [(s, len(s.get('history', []))) for s in self.data['sessions']]
        sessions_with_points.sort(key=lambda x: x[1])
        to_remove = len(self.data['sessions']) - PARAMS['max_sessions']
        remove_ids = [s[0]['id'] for s in sessions_with_points[:to_remove]]
        self.data['sessions'] = [s for s in self.data['sessions'] if s['id'] not in remove_ids]
        log.logger.info(f"🧹 Удалено {to_remove} пустых сессий")

    def check_new_sessions(self):
        """Проверка всех непроанализированных сессий"""
        analyzed = 0
        for session in self.data['sessions']:
            if session.get('analyzed', False):
                continue
            # Проверяем, можно ли анализировать (есть end или достаточно точек)
            has_end = 'end' in session
            points = len(session.get('history', []))
            if (has_end or points >= 5) and points >= 5:
                log.logger.info(f"🔍 Найдена непроанализированная сессия #{session['id']} с {points} точками")
                self._learn_from_session(session)
                session['analyzed'] = True
                analyzed += 1
        
        if analyzed > 0:
            self._save()
            log.logger.info(f"✅ Проанализировано новых сессий: {analyzed}")

    def start_session(self, target, bed, start_temp, start_type, ambient):
        """Начало новой сессии нагрева с проверкой новых сессий"""
        # При старте новой сессии проверяем, нет ли непроанализированных
        self.check_new_sessions()
        
        sid = len(self.data['sessions']) + 1
        self.data['sessions'].append({
            'id': sid,
            'start': time.time(),
            'target': target,
            'bed_target': bed,
            'ambient': ambient,
            'start_temp': start_temp,
            'start_type': start_type,
            'success': False,
            'history': [],
            'analyzed': False
        })
        self._cleanup_sessions()
        self._save()
        return sid

    def add_point(self, sid, elapsed, chamber, bed):
        for s in self.data['sessions']:
            if s['id'] == sid:
                s['history'].append({
                    't': round(elapsed, 1),
                    'chamber': round(chamber, 2),
                    'bed': round(bed, 2)
                })
                self._save()
                return

    def finish(self, sid, success, ttt=None):
        for s in self.data['sessions']:
            if s['id'] == sid:
                s['end'] = time.time()
                s['success'] = success
                if ttt:
                    s['time_to_target'] = round(ttt, 1)
                # Анализируем текущую сессию
                self._learn_from_session(s)
                s['analyzed'] = True
                self._save()
                return

    def _get_bed_rate_from_session(self, history, bed_target):
        if len(history) < 3:
            return None
        start_temp = history[0]['chamber']
        start_bed = history[0]['bed']
        rates = []
        for i in range(1, min(18, len(history))):
            if history[i]['chamber'] > start_temp + 0.5:
                break
            dt = history[i]['t'] - history[0]['t']
            if dt >= 30:
                rate = (history[i]['bed'] - start_bed) / (dt / 60)
                if 3.0 < rate < 20.0:
                    rates.append(rate)
        if rates:
            return np.median(rates)
        return None

    def _learn_from_session(self, session):
        history = session['history']
        points = len(history)
        if points < 5:
            log.logger.info(f"⏩ Сессия #{session['id']}: мало точек ({points})")
            return

        bed_rate = self._get_bed_rate_from_session(history, session['bed_target'])
        k_values = []
        bed_target = session['bed_target']
        rejected_k = 0

        for i in range(len(history) - 3):
            p1 = history[i]
            p2 = history[i + 2]
            dt = p2['t'] - p1['t']
            if dt < 20 or dt > 60:
                rejected_k += 1
                continue
            if abs(p2['bed'] - p1['bed']) > 5:
                rejected_k += 1
                continue
            
            bed = (p1['bed'] + p2['bed']) / 2
            if bed <= p1['chamber'] or bed <= p2['chamber']:
                rejected_k += 1
                continue
            
            ratio = (bed - p2['chamber']) / (bed - p1['chamber'])
            if not (0.01 < ratio < 0.99):
                rejected_k += 1
                continue
            
            try:
                new_k = -math.log(ratio) / dt
                if 0.00008 < new_k < 0.00035:
                    k_values.append(new_k)
                else:
                    rejected_k += 1
            except:
                rejected_k += 1

        found_k = len(k_values)
        log.session_analysis(session['id'], points, found_k, bed_rate if bed_rate else 0)

        if found_k > 0:
            median_k = np.median(k_values)
            rounded_new = round(median_k, 6)
            exists = False
            for existing in self.data['samples']:
                if abs(round(existing, 6) - rounded_new) < 0.000001:
                    exists = True
                    break
            if not exists:
                self.data['samples'].append(median_k)
                log.logger.info(f"✅ Сессия #{session['id']}: добавлен новый семпл k_heat = {median_k:.6f}")
                if len(self.data['samples']) > 500:
                    self.data['samples'] = self.data['samples'][-500:]
                self._recalibrate_k()
            else:
                log.logger.info(f"⏩ Сессия #{session['id']}: семпл {median_k:.6f} уже существует, пропускаем")

        if bed_rate:
            rounded_rate = round(bed_rate, 1)
            exists = False
            for existing in self.data['bed_rates']:
                if abs(round(existing, 1) - rounded_rate) < 0.1:
                    exists = True
                    break
            if not exists:
                self.data['bed_rates'].append(bed_rate)
                log.logger.info(f"✅ Сессия #{session['id']}: добавлена новая скорость стола = {bed_rate:.1f}°C/мин")
                if len(self.data['bed_rates']) > 100:
                    self.data['bed_rates'] = self.data['bed_rates'][-100:]
                
                old_rate = np.median(self.data['bed_rates'][-20:-1]) if len(self.data['bed_rates']) > 20 else 0
                new_rate = np.median(self.data['bed_rates'][-20:])
                if old_rate > 0 and abs(new_rate - old_rate) > 0.5:
                    log.bed_rate_update(old_rate, new_rate, len(self.data['bed_rates'][-20:]))
            else:
                log.logger.info(f"⏩ Сессия #{session['id']}: скорость {bed_rate:.1f}°C/мин уже существует, пропускаем")

    def _recalibrate_k(self):
        samples = self.data['samples']
        if len(samples) >= 15:
            unique_samples = []
            for s in samples:
                rounded = round(s, 6)
                if not any(abs(round(us, 6) - rounded) < 0.000001 for us in unique_samples):
                    unique_samples.append(s)
            
            recent_samples = unique_samples[-100:] if len(unique_samples) > 100 else unique_samples
            new_k = np.median(recent_samples)
            old_k = self.data['k_heat']
            
            self.data['k_heat'] = old_k * 0.9 + new_k * 0.1
            self.physics.k_heat = self.data['k_heat']
            log.k_heat_update(old_k, self.data['k_heat'], len(recent_samples))
            self._save()

    def get_history(self, sid, max_points=10):
        for s in self.data['sessions']:
            if s['id'] == sid:
                h = s['history'][-max_points:]
                return [p['t'] for p in h], [p['chamber'] for p in h]
        return [], []

    def maybe_log_sessions(self):
        current = len(self.data.get('sessions', []))
        if current != self.last_logged_sessions:
            log.logger.info(f"✅ Загружена база знаний v2.0, сессий: {current}")
            self.last_logged_sessions = current

# ==================== MOONRAKER ====================
def moonraker_get(path):
    try:
        return requests.get(f"http://127.0.0.1:7125{path}", timeout=3).json()
    except:
        return None

def send_vars(vars_dict):
    if not vars_dict:
        return
    script = "\n".join([
        f'SET_GCODE_VARIABLE MACRO="_VENT_VAR" VARIABLE="{k}" VALUE={v}'
        for k, v in vars_dict.items()
    ])
    
    if 'error_code' in vars_dict and vars_dict['error_code'] in [5, 6]:
        script += "\nVENT_STATUS"
        log.logger.info(f"🔔 Вызов VENT_STATUS для error_code={vars_dict['error_code']}")
    
    try:
        requests.post("http://127.0.0.1:7125/printer/gcode/script",
                      json={"script": script}, timeout=3)
    except:
        pass

# ==================== СОСТОЯНИЕ ====================
@dataclass
class State:
    session_id: Optional[int] = None
    call_count: int = 0
    kalman: Optional[Dict] = None
    last_pred: int = 0
    last_rate: float = 0.0
    cooldown_start: float = 0
    last_adapt: float = 0
    error: int = 0
    rate_history: deque = field(default_factory=lambda: deque(maxlen=PARAMS['rate_history_size']))
    startup_logged: bool = False
    target_reached_internal: bool = False
    session_closed: bool = False
    waiting_for_external: bool = False
    last_heating_valid: int = 0
    last_sent_pred: int = 0
    last_sent_rate: float = 0.0
    stable_since: float = 0
    stabilization_logged: bool = False
    initial_target: float = 0.0
    first_session: bool = True

    def add_rate(self, rate):
        self.rate_history.append(rate)

    def all_rates_low(self, threshold=0.05):
        if not self.rate_history:
            return False
        return all(r < threshold for r in self.rate_history)

    def update_cooldown(self, rate, now):
        if rate < 0.01:
            if self.cooldown_start == 0:
                self.cooldown_start = now
        else:
            self.cooldown_start = 0

    def can_adapt(self, now, min_interval=180):
        return now - self.last_adapt > min_interval

    def update_stable(self, is_stable, now):
        if is_stable:
            if self.stable_since == 0:
                self.stable_since = now
                self.stabilization_logged = False
        else:
            self.stable_since = 0
            self.stabilization_logged = False

    def is_ready_for_adaptation(self, now):
        if self.stable_since == 0:
            return False
        return now - self.stable_since >= PARAMS['stabilization_delay']

    def get_target_time(self, elapsed, time_left):
        if elapsed < self.initial_target:
            return self.initial_target, True
        else:
            return time_left / 2, False

def load_state():
    state = State()
    if not os.path.exists(STATE_PATH):
        return state
    try:
        with open(STATE_PATH, 'r') as f:
            content = f.read().strip()
            if not content:
                return state
            data = json.loads(content)
            for k, v in data.items():
                if hasattr(state, k):
                    if k == 'rate_history' and v:
                        state.rate_history = deque(v, maxlen=PARAMS['rate_history_size'])
                    else:
                        setattr(state, k, v)
        return state
    except Exception:
        return state

def save_state(state):
    try:
        state_dict = asdict(state)
        state_dict['rate_history'] = list(state.rate_history)
        with open(STATE_PATH + '.tmp', 'w') as f:
            json.dump(state_dict, f)
        os.replace(STATE_PATH + '.tmp', STATE_PATH)
    except Exception:
        pass

# ==================== ОСНОВНАЯ ЛОГИКА ====================
def process_heating(input_vars):
    start = time.time()
    kb = KnowledgeBase()
    state = load_state()

    control = int(input_vars.get("control_mode", 0))
    target_reached = int(input_vars.get("target_reached", 0))
    wait_start = int(input_vars.get("wait_start_loop", -1))
    last_time = float(input_vars.get("last_time_for_rate", 0))
    loop = int(input_vars.get("loop_counter", 0))
    elapsed = (loop - wait_start) * 10.0 if wait_start >= 0 else 0

    if last_time == -1.0 and control != 3:
        state = State()
        save_state(state)
        return {"last_time_for_rate": 1.0}

    chamber_target = float(input_vars.get("chamber_target", 40))
    bed_target = float(input_vars.get("bed_target_temp", 50))
    ambient = float(input_vars.get("start_chamber_temp", 25))
    bed_min = float(input_vars.get("bed_min_temp", 50))
    real_bed = float(input_vars.get("real_bed_temp", 0))
    current = float(input_vars.get("ema_chamber_temp", 25))
    median = float(input_vars.get("current_median_temp", current))
    time_left = float(input_vars.get("time_left_sec", 0)) / 60

    # ========== ОБРАБОТКА ДОСТИЖЕНИЯ ЦЕЛИ (УПРОЩЕННАЯ) ==========

    # 1. Цель достигнута (внутренне или внешне) - закрываем сессию
    if not state.target_reached_internal and state.session_id:
        # Проверяем оба условия: внутреннее ИЛИ внешнее
        if (current >= chamber_target - PARAMS['target_hysteresis']) or (target_reached == 1):
            reason = "внешний флаг" if target_reached == 1 else "цель достигнута"
            log.target_reached(current, chamber_target, state.session_id)
            state.target_reached_internal = True
            state.waiting_for_external = (target_reached != 1)  # Ждем внешний флаг, если его еще нет
            
            if not state.session_closed:
                log.session_closed(state.session_id, reason)
                kb.finish(state.session_id, True, elapsed)
                state.session_closed = True
            
            save_state(state)
            return {}

    # 2. Ждем внешнего подтверждения (если нужно)
    if state.target_reached_internal and state.waiting_for_external:
        if target_reached == 1:
            log.external_target_received(state.session_id)
            state.waiting_for_external = False
            save_state(state)
        return {}  # Всегда выходим

    # 3. Ждем остывания (после получения внешнего флага)
    if state.target_reached_internal and not state.waiting_for_external:
        if target_reached == 0:
            log.cooldown_ready()
            new_state = State()
            new_state.startup_logged = state.startup_logged
            new_state.first_session = False
            save_state(new_state)
        return {}  # Всегда выходим

    # 4. Принудительное выключение
    if wait_start == -1 and state.session_id and not state.session_closed:
        log.logger.info(f"🔌 VENT_OFF - сес #{state.session_id} завершена")
        kb.finish(state.session_id, False)
        state = State()
        save_state(state)
        return {}

    if wait_start == -1 and not state.session_id:
        return {}

    # ========== ИНИЦИАЛИЗАЦИЯ НОВОЙ СЕССИИ ==========
    if not state.kalman:
        if not kb.physics.is_reachable(ambient, chamber_target):
            k_eq = kb.physics.get_k_eq()
            max_eq = ambient + k_eq * (PARAMS['max_bed'] - ambient)
            log.logger.warning(f"⚠️ цель {chamber_target}°C недостиж (max {max_eq:.1f}°C)")
            return {"error_code": 6, "equilibrium_est": max_eq}

        if time_left > 0:
            target_time = time_left / 2
            state.initial_target = target_time
            
            if state.first_session:
                start_temp = ambient
                start_type = "ambient"
            else:
                start_temp = current
                start_type = "ema"

            bed_needed = kb.physics.bed_needed(start_temp, chamber_target, target_time, bed_min)
            bed_rate = kb.get_bed_rate()
            
            if bed_rate > 0 and real_bed < bed_needed:
                heat_time = (bed_needed - real_bed) / bed_rate
                adjusted_target_time = max(target_time - heat_time, target_time * 0.7)
                if adjusted_target_time < target_time:
                    bed_needed = kb.physics.bed_needed(start_temp, chamber_target, adjusted_target_time, bed_min)
            
            bed_init = min(max(bed_needed, bed_min, real_bed), PARAMS['max_bed'])
            init_pred = kb.physics.time_to_target(bed_init, start_temp, chamber_target)
            heat_time = (bed_init - real_bed) / bed_rate if bed_rate > 0 and bed_init > real_bed else 0
        else:
            bed_init = max(bed_min, real_bed)
            init_pred = None
            heat_time = 0
            bed_rate = kb.get_bed_rate()
            start_temp = current
            start_type = "ema"

        state.session_id = kb.start_session(chamber_target, bed_init, start_temp, start_type, ambient)
        state.kalman = {"x": [start_temp, 0.0], "P": [[100, 0], [0, 100]]}
        
        if not state.startup_logged:
            log.startup(kb.physics.k_heat, kb.get_samples_count(), bed_rate)
            log.status(len(kb.data.get('sessions', [])), kb.get_samples_count(), bed_rate)
            state.startup_logged = True
        
        save_state(state)
        log.init(bed_init, kb.physics.k_heat, start_temp, start_type, init_pred or 0, bed_rate, heat_time)
        return {"bed_target_temp": round(bed_init, 0)}

    # ========== ОСНОВНОЙ ЦИКЛ ==========
    kf = KalmanFilter(0)
    kf.x = np.array(state.kalman['x'])
    kf.P = np.array(state.kalman['P'])
    filtered, display_rate = kf.update(median)
    state.kalman = {"x": kf.x.tolist(), "P": kf.P.tolist()}
    
    state.add_rate(display_rate)
    state.last_rate = display_rate
    state.call_count += 1

    if state.session_id:
        kb.add_point(state.session_id, elapsed, median, real_bed)

    bed_stable = real_bed >= bed_target - PARAMS['bed_stable']
    state.update_stable(bed_stable, elapsed)

    if bed_stable and state.stable_since > 0 and not state.stabilization_logged:
        log.stabilization_start(state.session_id)
        state.stabilization_logged = True

    if state.is_ready_for_adaptation(elapsed) and state.stable_since > 0:
        log.stabilization_ready(state.session_id, elapsed - state.stable_since)

    state.update_cooldown(display_rate, elapsed)
    cooldown_time = state.cooldown_start and (elapsed - state.cooldown_start)

    pred = kb.physics.time_to_target(bed_target, filtered, chamber_target)
    method = "physical" if pred else None
    
    if pred is not None:
        pred_int = int(round(pred))
        state.last_pred = pred_int
        heating_estimate_valid = 1
    else:
        pred_int = state.last_pred
        heating_estimate_valid = 0

    current_error = int(input_vars.get("error_code", 0))
    result = {}

    if pred_int != state.last_sent_pred:
        result["last_target_minutes"] = pred_int
        state.last_sent_pred = pred_int

    rounded_rate = round(display_rate, 2)
    if abs(rounded_rate - state.last_sent_rate) > 0.01:
        result["current_heat_rate_cpm"] = rounded_rate
        state.last_sent_rate = rounded_rate

    if heating_estimate_valid != state.last_heating_valid:
        result["heating_estimate_valid"] = heating_estimate_valid
        state.last_heating_valid = heating_estimate_valid

    target_time, is_initial_target = state.get_target_time(elapsed, time_left)

    if pred is not None:
        log.predict_full(pred_int, elapsed, time_left, target_time, method, is_initial_target)

    new_bed = bed_target
    adapted = False

    # === ОПТИМИЗАЦИЯ: единый расчет bed_needed ===
    if (state.is_ready_for_adaptation(elapsed) and
        pred is not None and
        elapsed >= PARAMS['adapt_start'] and
        state.can_adapt(elapsed)):
        
        stable_since = elapsed - state.stable_since
        bed_needed_for_target = kb.physics.bed_needed(filtered, chamber_target, target_time, bed_min)
        
        if bed_needed_for_target > bed_target + 5:
            new_bed = min(round(bed_needed_for_target / 5) * 5, PARAMS['max_bed'])
            new_bed = max(new_bed, bed_min)
            adapted = True
            state.last_adapt = elapsed
            log.adapt_up(bed_target, new_bed, pred_int, target_time, stable_since, is_initial_target)
            
        elif (bed_target > bed_needed_for_target + 10 and
              elapsed < state.initial_target):
            new_bed = max(round(bed_needed_for_target / 5) * 5, bed_min)
            adapted = True
            state.last_adapt = elapsed
            log.adapt_down(bed_target, new_bed, pred_int, target_time, stable_since, is_initial_target)

    # Принудительная адаптация при застое
    if not adapted and bed_stable and cooldown_time and cooldown_time >= PARAMS['cooldown_sec']:
        if state.all_rates_low(0.05):
            if state.can_adapt(elapsed, 120):
                new_bed = min(bed_target + PARAMS['adapt_step'], PARAMS['max_bed'])
                new_bed = max(new_bed, bed_min)
                adapted = True
                state.last_adapt = elapsed
                log.adapt_force(bed_target, new_bed, f"cooldown {cooldown_time:.0f}с, скорость {display_rate:.2f}°C/мин")

    if adapted:
        result["bed_target_temp"] = round(new_bed, 0)

    # Обработка ошибки 5 (Таймаут прогноза)
    error_to_send = current_error
    if pred is not None and time_left > 0 and bed_stable:
        if (pred - PARAMS['error5_margin']) > time_left:
            if current_error != 5:
                error_to_send = 5
                log.error_5_set(pred, time_left)
        elif current_error == 5 and pred <= time_left * 0.8:
            error_to_send = 0
            log.error_5_cleared(pred, time_left)

    if error_to_send != current_error:
        result["error_code"] = error_to_send

    samples = kb.get_samples_count()
    bed_rate = kb.get_bed_rate()
    log.result(filtered, display_rate, result.get('bed_target_temp', bed_target), real_bed, bed_stable)
    
    save_state(state)
    return result

# ==================== MAIN ====================
lock = threading.RLock()

def main():
    kb = KnowledgeBase()
    kb.maybe_log_sessions()
    
    obj = moonraker_get("/printer/objects/query?gcode_macro%20_VENT_VAR")
    if not obj:
        log.logger.error("moonraker unreachable")
        return
    
    macro = obj.get('result', {}).get('status', {}).get('gcode_macro _VENT_VAR', {})
    if not macro:
        log.logger.error("_VENT_VAR not found")
        return

    keys = ['control_mode', 'target_reached', 'wait_start_loop', 'last_time_for_rate',
            'loop_counter', 'current_median_temp', 'chamber_target', 'bed_target_temp',
            'start_chamber_temp', 'bed_min_temp', 'real_bed_temp', 'time_left_sec',
            'ema_chamber_temp', 'error_code']
    
    input_vars = {k: macro.get(k) for k in keys if k in macro}
    
    required = ['chamber_target', 'ema_chamber_temp', 'loop_counter', 'real_bed_temp']
    for key in required:
        if key not in input_vars:
            log.logger.error(f"missing required key: {key}")
            return

    result = process_heating(input_vars)
    
    if result:
        log.logger.info(f"📤 отправка: {list(result.keys())}")
        send_vars(result)

if __name__ == "__main__":
    main()
    
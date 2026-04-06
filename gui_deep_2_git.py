#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')  # Definir backend antes de outros imports

import tkinter as tk
from tkinter import ttk, messagebox
import serial
import numpy as np
import pylab as pl
import struct
from serial.tools import list_ports
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gc
import random

import os
from scipy.optimize import curve_fit
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

VID = 0x0483 #1155
PID = 0x5740 #22336

# Get nanovna device automatically
def getport() -> str:
    device_list = list_ports.comports()
    for device in device_list:
        if device.vid == VID and device.pid == PID:
            return device.device
    raise OSError("device not found")

REF_LEVEL = (1<<9)

class NanoVNA:
    def __init__(self, dev = None):
        self.dev = dev or getport()
        self.serial = None
        self._frequencies = None
        self.points = 2020
        
    @property
    def frequencies(self):
        return self._frequencies
    
    def resume(self):
        self.send_command("resume\r")

    def set_frequencies(self, start = None, stop = None, points = 101):
        if points:
            self.points = points
        self._frequencies = np.linspace(start, stop, self.points)

    def open(self):
        if self.serial is None:
            self.serial = serial.Serial(self.dev)

    def close(self):
        if self.serial:
            self.serial.close()
        self.serial = None

    def send_command(self, cmd):
        self.open()
        self.serial.write(cmd.encode())
        self.serial.readline() # discard empty line

    def set_sweep(self, start, stop):
        if start is not None:
            self.send_command("sweep start %d\r" % start)
        if stop is not None:
            self.send_command("sweep stop %d\r" % stop)

    def set_frequency(self, freq):
        if freq is not None:
            self.send_command("freq %d\r" % freq)

    def fetch_data(self):
        result = ''
        line = ''
        while True:
            c = self.serial.read().decode('utf-8')
            if c == chr(13):
                next # ignore CR
            line += c
            if c == chr(10):
                result += line
                line = ''
                next
            if line.endswith('ch>'):
                # stop on prompt
                break
        return result

    def fetch_buffer(self, freq = None, buffer = 0):
        self.send_command("dump %d\r" % buffer)
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.extend([int(d, 16) for d in line.strip().split(' ')])
        return np.array(x, dtype=np.int16)

    def fetch_rawwave(self, freq = None):
        if freq:
            self.set_frequency(freq)
            time.sleep(0.05)
        self.send_command("dump 0\r")
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.extend([int(d, 16) for d in line.strip().split(' ')])
        return np.array(x[0::2], dtype=np.int16), np.array(x[1::2], dtype=np.int16)

    def fetch_array(self, sel):
        self.send_command("data %d\r" % sel)
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.extend([float(d) for d in line.strip().split(' ')])
                
        return np.array(x[0::2]) + np.array(x[1::2]) * 1j
    

    def data(self, array = 0):
        self.send_command("data %d\r" % array)
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                d = line.strip().split(' ')
                x.append(float(d[0])+float(d[1])*1.j)
                #time.sleep(0.1)                
        return np.array(x)

    def fetch_frequencies(self):
        self.send_command("frequencies\r")
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.append(float(line))
        self._frequencies = np.array(x)
        return x

    def send_scan(self, start = None, stop = None, points = 101):
        if points:
            self.send_command("scan %d %d %d\r"%(start, stop, points))
        else:
            self.send_command("scan %d %d\r"%(start, stop))

    def scan(self):
        segment_length = 101
        array0 = []
        s11 = []
        array1 = []
        if self._frequencies is None:
            self.fetch_frequencies()
        freqs = self._frequencies
        while len(freqs) > 0:
            seg_start = freqs[0]
            seg_stop = freqs[segment_length-1] if len(freqs) >= segment_length else freqs[-1]
            length = segment_length if len(freqs) >= segment_length else len(freqs)
            #print((seg_start, seg_stop, length))
            self.send_scan(seg_start, seg_stop, length)
            array0.extend(self.data(1))
            s11.extend(self.data(1))
            array1.extend(self.data(1))
            freqs = freqs[segment_length:]
            
        self.resume()
        return (s11)
    
            
    def clear_buffers(self):
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

class NanoVNAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NanoVNA Control Interface")
        self.root.geometry("1400x900")
        
        # Variáveis de controle
        self.start_freq = tk.DoubleVar(value=6.1e6)  # Valor inicial padrão
        self.stop_freq = tk.DoubleVar(value=6.16e6)  # Valor final padrão
        self.num_points = tk.IntVar(value=1010)      # Número de pontos padrão
        self.is_measuring = False
        self.parameters_updated = False
        self.file_counter = 1  # Contador para numeração de arquivos
        
        # Inicializar nanoVNA
        self.nvna = NanoVNA()
        
        # Dados para os gráficos
        self.times = []
        self.freq_resonance = []
        self.dissipation = []
        self.freq_res_ajustada = []
        self.dis_ajustada = []
        
        # Dados para o gráfico de condutância
        self.last_freqs = []
        self.last_condutance = []
        self.last_fit = []
        
        # Configurar interface
        self.create_widgets()
        self.create_graphs()
        
        # Inicializar contador de arquivos
        self.initialize_file_counter()
        
    def initialize_file_counter(self):
        """Inicializa o contador de arquivos baseado nos arquivos existentes"""
        diretorio = "***************your directory path here******************"
        
        # Verificar todos os arquivos CSV no diretório
        if os.path.exists(diretorio):
            arquivos = [f for f in os.listdir(diretorio) if f.endswith('.csv')]
            numeros = []
            
            for arquivo in arquivos:
                # Extrair número do nome do arquivo
                partes = arquivo.split('.')[0].split('_')
                if len(partes) > 1 and partes[-1].isdigit():
                    numeros.append(int(partes[-1]))
            
            if numeros:
                self.file_counter = max(numeros) + 1
                
    def get_next_filename(self, prefixo, extensao=".csv"):
        """Gera o próximo nome de arquivo com numeração sequencial"""
        diretorio = "***************your directory path here******************"
        nome_arquivo = f"{prefixo}_{self.file_counter}{extensao}"
        caminho_completo = os.path.join(diretorio, nome_arquivo)
        return caminho_completo
        
    def create_widgets(self):
        # Frame principal para controles
        main_control_frame = ttk.Frame(self.root)
        main_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Frame para parâmetros
        control_frame = ttk.LabelFrame(main_control_frame, text="Parâmetros da NanoVNA")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Entrada para frequência inicial
        ttk.Label(control_frame, text="Frequência inicial/Hz:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        freq_start_entry = ttk.Entry(control_frame, textvariable=self.start_freq, width=15)
        freq_start_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Entrada para frequência final
        ttk.Label(control_frame, text="Frequência final/Hz:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        freq_stop_entry = ttk.Entry(control_frame, textvariable=self.stop_freq, width=15)
        freq_stop_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Entrada para número de pontos
        ttk.Label(control_frame, text="Número de pontos:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        points_entry = ttk.Entry(control_frame, textvariable=self.num_points, width=15)
        points_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Botão para atualizar parâmetros
        ttk.Button(control_frame, text="Atualizar Parâmetros", 
                  command=self.update_parameters).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Botão para iniciar/parar medição
        self.start_button = ttk.Button(control_frame, text="Iniciar Medição", 
                                      command=self.toggle_measurement)
        self.start_button.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Parâmetros não atualizados")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Botão para salvar dados
        ttk.Button(control_frame, text="Salvar Dados", 
                  command=self.save_data).grid(row=6, column=0, columnspan=2, pady=10)
        
        # Label para mostrar o número do arquivo atual
        self.file_counter_label = ttk.Label(control_frame, text=f"Próximo arquivo: {self.file_counter}")
        self.file_counter_label.grid(row=7, column=0, columnspan=2, pady=5)
        
        # Frame para monitor de valores
        monitor_frame = ttk.LabelFrame(main_control_frame, text="Monitor de Valores")
        monitor_frame.pack(fill=tk.X, pady=5)
        
        # Labels para mostrar valores atuais
        ttk.Label(monitor_frame, text="Frequência de Ressonância:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.lbl_freq_res = ttk.Label(monitor_frame, text="-")
        self.lbl_freq_res.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(monitor_frame, text="Frequência Ajustada:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.lbl_freq_ajust = ttk.Label(monitor_frame, text="-")
        self.lbl_freq_ajust.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(monitor_frame, text="Dissipação:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.lbl_diss = ttk.Label(monitor_frame, text="-")
        self.lbl_diss.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(monitor_frame, text="Dissipação Ajustada:", font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.lbl_diss_ajust = ttk.Label(monitor_frame, text="-")
        self.lbl_diss_ajust.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        
    def create_graphs(self):
        # Frame para os gráficos
        graph_frame = ttk.Frame(self.root)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Notebook para abas de gráficos
        self.notebook = ttk.Notebook(graph_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba para gráficos de tempo
        time_frame = ttk.Frame(self.notebook)
        self.notebook.add(time_frame, text="Tempo Real")
        
        # Aba para gráfico de condutância
        cond_frame = ttk.Frame(self.notebook)
        self.notebook.add(cond_frame, text="Condutância vs Frequência")
        
        # Criar gráficos de tempo
        self.fig_time = Figure(figsize=(8, 6), dpi=100)
        self.ax1 = self.fig_time.add_subplot(211)  # Primeiro gráfico (superior)
        self.ax2 = self.fig_time.add_subplot(212)  # Segundo gráfico (inferior)
        
        # Canvas para gráficos de tiempo
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=time_frame)
        self.canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configurações dos gráficos de tempo
        self.ax1.set_title("Frequência de Ressonância vs Tempo")
        self.ax1.set_xlabel("Tempo (s)")
        self.ax1.set_ylabel("Frequência (Hz)")
        self.ax1.grid(True)
        
        self.ax2.set_title("Dissipação vs Tempo")
        self.ax2.set_xlabel("Tempo (s)")
        self.ax2.set_ylabel("Dissipação")
        self.ax2.grid(True)
        
        # Criar gráfico de condutância
        self.fig_cond = Figure(figsize=(8, 6), dpi=100)
        self.ax_cond = self.fig_cond.add_subplot(111)
        
        # Canvas para gráfico de condutância
        self.canvas_cond = FigureCanvasTkAgg(self.fig_cond, master=cond_frame)
        self.canvas_cond.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configurações do gráfico de condutância
        self.ax_cond.set_title("Condutância vs Frequência")
        self.ax_cond.set_xlabel("Frequência (Hz)")
        self.ax_cond.set_ylabel("Condutância")
        self.ax_cond.grid(True)
        
    def update_parameters(self):
        try:
            # Validar entradas
            start = self.start_freq.get()
            stop = self.stop_freq.get()
            points = self.num_points.get()
            
            if start <= 0 or stop <= 0 or start >= stop:
                messagebox.showerror("Erro", "Valores de frequência inválidos!")
                return
                
            if points <= 0:
                messagebox.showerror("Erro", "Número de pontos deve ser positivo!")
                return
                
            # Atualizar status
            self.parameters_updated = True
            self.status_label.config(text="Parâmetros atualizados com sucesso!")
            
            # Limpar dados anteriores
            self.times = []
            self.freq_resonance = []
            self.dissipation = []
            self.freq_res_ajustada = []
            self.dis_ajustada = []
            
            # Atualizar gráficos vazios
            self.update_plots()
            
        except ValueError:
            messagebox.showerror("Erro", "Valores inválidos inseridos!")
            
    def toggle_measurement(self):
        if not self.parameters_updated:
            messagebox.showwarning("Aviso", "Atualize os parâmetros primeiro!")
            return
            
        if not self.is_measuring:
            # Iniciar medição
            self.is_measuring = True
            self.start_button.config(text="Parar Medição")
            self.measurement_thread = threading.Thread(target=self.run_measurement)
            self.measurement_thread.daemon = True
            self.measurement_thread.start()
        else:
            # Parar medição
            self.is_measuring = False
            self.start_button.config(text="Iniciar Medição")
            
    def run_measurement(self):
        # Inicializar variáveis de medição
        pi_np = np.pi
        def lorentziana(x, x0, gamma, B, C):
            return (B + ((2*C * gamma/pi_np) / (4*(x - x0)**2 + gamma**2)) )
        
        try:
            self.nvna.open()
            passed_seconds = 0
            
            # Obter parâmetros da interface
            freq_inicial = self.start_freq.get()
            freq_final = self.stop_freq.get()
            numpontos = self.num_points.get()
            numvarreduras = numpontos/101
            int_numvrreduras = int(numvarreduras)
            step = (freq_final - freq_inicial)/(numpontos-1)
            
            while self.is_measuring:
                begin_time = time.time()
                fi = freq_inicial
                total_s11 = np.array([])
                freqs_total = []
                
                for n in range(1, int_numvrreduras+1):
                    ff = fi + (100*step)
                    self.nvna.set_sweep(fi, ff)
                    self.nvna.set_frequencies(start=fi, stop=ff, points=101)
                    freqs = self.nvna.fetch_frequencies()
                    self.nvna.send_scan(start=fi, stop=ff, points=101)
                    s11 = self.nvna.data(0)
                    
                    total_s11 = np.append(total_s11, s11)
                    freqs_total.extend(freqs)
                    fi = ff + step
                
                # Processar dados
                real_total_s11 = total_s11.real
                yn = (1-total_s11)/(1+total_s11)
                condutancia = yn.real
                conductancia_max = np.max(condutancia)
                index_max = np.argmax(condutancia)
                frequencia_ressonancia = freqs_total[index_max]
                meia_altura = conductancia_max/2
                indices_meia_altura = np.where(condutancia >= meia_altura)[0]
                inicio_banda = indices_meia_altura[0]
                fim_banda = indices_meia_altura[-1]
                largura_banda_meia_altura = freqs_total[fim_banda] - freqs_total[inicio_banda]
                dissipação = largura_banda_meia_altura/frequencia_ressonancia
                
                # Ajuste de curva
                condutancia_int = condutancia
                freqs_total_int = np.array(freqs_total)
                parametros_iniciais = [frequencia_ressonancia, largura_banda_meia_altura, inicio_banda, 1000]
                param_otimos, covariancia = curve_fit(lorentziana, freqs_total_int, condutancia_int, p0=parametros_iniciais)
                x0_ajustado, gamma_ajustado, B_ajustado, C_ajustado = param_otimos
                dissipação_ajustada = abs(gamma_ajustado)/x0_ajustado
                
                # Armazenar dados para o gráfico de condutância
                self.last_freqs = freqs_total_int
                self.last_condutance = condutancia_int
                self.last_fit = lorentziana(freqs_total_int, *param_otimos)
                
                # Atualizar dados de tempo
                end_time = time.time()
                passed_seconds += (end_time - begin_time)
                
                # Adicionar aos dados dos gráficos
                self.times.append(passed_seconds)
                self.freq_resonance.append(frequencia_ressonancia)
                self.dissipation.append(dissipação)
                self.freq_res_ajustada.append(x0_ajustado)
                self.dis_ajustada.append(dissipação_ajustada)
                
                # Atualizar gráficos
                self.root.after(0, self.update_plots)
                
                # Limpar buffers e preparar para próxima iteração
                self.nvna.clear_buffers()
                time.sleep(0.1)  # Pequena pausa entre medições
                
        except Exception as e:
            messagebox.showerror("Erro", f"Falha na medição: {str(e)}")
            self.is_measuring = False
            self.start_button.config(text="Iniciar Medição")
        finally:
            self.nvna.close()
            
    def update_plots(self):
        # Atualizar primeiro gráfico (Frequência de Ressonância)
        self.ax1.clear()
        if len(self.times) > 0:
            self.ax1.plot(self.times, self.freq_resonance, 'b-', label='Frequência de Ressonância')
            self.ax1.plot(self.times, self.freq_res_ajustada, 'r-', label='Frequência Ajustada')
        self.ax1.set_title("Frequência de Ressonância vs Tempo")
        self.ax1.set_xlabel("Tempo (s)")
        self.ax1.set_ylabel("Frequência (Hz)")
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Atualizar segundo gráfico (Dissipação)
        self.ax2.clear()
        if len(self.times) > 0:
            self.ax2.plot(self.times, self.dissipation, 'b-', label='Dissipação')
            self.ax2.plot(self.times, self.dis_ajustada, 'r-', label='Dissipação Ajustada')
        self.ax2.set_title("Dissipação vs Tempo")
        self.ax2.set_xlabel("Tempo (s)")
        self.ax2.set_ylabel("Dissipação")
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Atualizar gráfico de condutância
        self.ax_cond.clear()
        if len(self.last_freqs) > 0:
            self.ax_cond.plot(self.last_freqs, self.last_condutance, 'b-', label='Dados Experimentais')
            self.ax_cond.plot(self.last_freqs, self.last_fit, 'r-', label='Curva Ajustada')
        self.ax_cond.set_title("Condutância vs Frequência")
        self.ax_cond.set_xlabel("Frequência (Hz)")
        self.ax_cond.set_ylabel("Condutância")
        self.ax_cond.grid(True)
        self.ax_cond.legend()
        
        # Atualizar monitor de valores
        if len(self.times) > 0:
            self.lbl_freq_res.config(text=f"{self.freq_resonance[-1]:.2f} Hz")
            self.lbl_freq_ajust.config(text=f"{self.freq_res_ajustada[-1]:.2f} Hz")
            self.lbl_diss.config(text=f"{self.dissipation[-1]:.6f}")
            self.lbl_diss_ajust.config(text=f"{self.dis_ajustada[-1]:.6f}")
        
        # Ajustar layout
        self.fig_time.tight_layout(pad=3.0)
        self.fig_cond.tight_layout(pad=3.0)
        
        # Redesenhar canvas
        self.canvas_time.draw()
        self.canvas_cond.draw()
        
    def save_data(self):
        # Salvar dados em arquivos CSV com numeração sequencial
        try:
            # Gerar nomes de arquivo com numeração sequencial
            arquivo_tempo = self.get_next_filename("tempo")
            arquivo_freq_ressonancia = self.get_next_filename("freq_ressonancia")
            arquivo_dissipacao = self.get_next_filename("dissipacao")
            arquivo_freq_ajustada = self.get_next_filename("freq_ajustada")
            arquivo_dissipacao_ajustada = self.get_next_filename("dissipacao_ajustada")
            
            # Salvar tempos
            with open(arquivo_tempo, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows([[t] for t in self.times])
            
            # Salvar frequências de ressonância
            with open(arquivo_freq_ressonancia, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows([[f] for f in self.freq_resonance])
            
            # Salvar dissipações
            with open(arquivo_dissipacao, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows([[d] for d in self.dissipation])
            
            # Salvar frequências ajustadas
            with open(arquivo_freq_ajustada, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows([[f] for f in self.freq_res_ajustada])
            
            # Salvar dissipações ajustadas
            with open(arquivo_dissipacao_ajustada, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows([[d] for d in self.dis_ajustada])
            
            # Salvar dados de condutância e frequências em arquivos separados
            if len(self.last_freqs) > 0:
                # Salvar apenas condutância
                arquivo_condutancia = self.get_next_filename("condutancia")
                with open(arquivo_condutancia, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows([[c] for c in self.last_condutance])
                
                # Salvar apenas frequências
                arquivo_frequencias = self.get_next_filename("frequencias_varredura")
                with open(arquivo_frequencias, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows([[f] for f in self.last_freqs])
                
                # Salvar dados combinados (frequência, condutância, ajuste)
                arquivo_combinado = self.get_next_filename("condutancia_freq_ajuste")
                with open(arquivo_combinado, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Frequência (Hz)", "Condutância", "Ajuste"])
                    writer.writerows(zip(self.last_freqs, self.last_condutance, self.last_fit))
            
            messagebox.showinfo("Sucesso", f"Dados salvos com sucesso! (Arquivo #{self.file_counter})")
            
            # Incrementar contador para próxima vez
            self.file_counter += 1
            self.file_counter_label.config(text=f"Próximo arquivo: {self.file_counter}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar dados: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NanoVNAApp(root)
    root.mainloop()
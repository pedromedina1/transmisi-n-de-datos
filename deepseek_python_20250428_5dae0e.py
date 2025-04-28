import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class SistemaTelecom:
    def __init__(self, tasa_bits=1000, frecuencia_portadora=5000, frecuencia_muestreo=44100):
        """
        Inicializa el sistema de telecomunicaciones con parámetros básicos.
        
        Args:
            tasa_bits (int): Tasa de bits en bps
            frecuencia_portadora (int): Frecuencia de la portadora en Hz
            frecuencia_muestreo (int): Frecuencia de muestreo en Hz
        """
        self.tasa_bits = tasa_bits
        self.fc = frecuencia_portadora
        self.fs = frecuencia_muestreo
        self.duracion_bit = 1 / tasa_bits
        self.muestras_por_bit = int(frecuencia_muestreo / tasa_bits)
        
    def generar_bits(self, num_bits=10):
        """Genera una secuencia aleatoria de bits."""
        self.bits = np.random.randint(0, 2, num_bits)
        return self.bits
    
    def modulacion_ask(self, bits, amplitud=1):
        """Modulación por desplazamiento de amplitud (ASK)."""
        t = np.linspace(0, len(bits)*self.duracion_bit, len(bits)*self.muestras_por_bit, endpoint=False)
        señal_modulada = np.zeros_like(t)
        
        for i, bit in enumerate(bits):
            inicio = i * self.muestras_por_bit
            fin = (i+1) * self.muestras_por_bit
            if bit == 1:
                señal_modulada[inicio:fin] = amplitud * np.sin(2*np.pi*self.fc*t[inicio:fin])
        
        self.señal_modulada = señal_modulada
        return señal_modulada, t
    
    def agregar_ruido(self, señal, snr_db=10):
        """Agrega ruido gaussiano blanco a la señal."""
        potencia_señal = np.mean(señal**2)
        potencia_ruido = potencia_señal / (10 ** (snr_db / 10))
        ruido = np.random.normal(0, np.sqrt(potencia_ruido), len(señal))
        señal_ruidosa = señal + ruido
        self.señal_ruidosa = señal_ruidosa
        return señal_ruidosa
    
    def demodulacion_ask(self, señal_ruidosa, umbral=0.5):
        """Demodulación ASK con detección por envolvente."""
        # Rectificar la señal
        señal_rectificada = np.abs(señal_ruidosa)
        
        # Filtro pasa bajos
        b, a = signal.butter(5, self.fc/(self.fs/2), btype='low')
        señal_filtrada = signal.lfilter(b, a, señal_rectificada)
        
        # Muestreo en el punto medio de cada bit
        puntos_muestreo = (np.arange(len(self.bits)) + 0.5) * self.muestras_por_bit
        puntos_muestreo = puntos_muestreo.astype(int)
        muestras = señal_filtrada[puntos_muestreo]
        
        # Decisión por umbral
        bits_recuperados = (muestras > umbral).astype(int)
        
        self.bits_recuperados = bits_recuperados
        return bits_recuperados
    
    def calcular_ber(self):
        """Calcula la tasa de error de bits (BER)."""
        if not hasattr(self, 'bits_recuperados'):
            raise ValueError("Primero debe demodular la señal")
        errores = np.sum(self.bits != self.bits_recuperados)
        return errores / len(self.bits)
    
    def graficar_señales(self, t, señal_modulada, señal_ruidosa):
        """Grafica las señales modulada y con ruido."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(t, señal_modulada)
        plt.title('Señal Modulada ASK')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        
        plt.subplot(2, 1, 2)
        plt.plot(t, señal_ruidosa)
        plt.title('Señal con Ruido')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        
        plt.tight_layout()
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear sistema de telecomunicaciones
    sistema = SistemaTelecom(tasa_bits=1000, frecuencia_portadora=5000)
    
    # Generar bits aleatorios
    bits = sistema.generar_bits(10)
    print("Bits transmitidos:", bits)
    
    # Modulación ASK
    señal_modulada, t = sistema.modulacion_ask(bits)
    
    # Agregar ruido
    señal_ruidosa = sistema.agregar_ruido(señal_modulada, snr_db=5)
    
    # Demodulación
    bits_recuperados = sistema.demodulacion_ask(señal_ruidosa)
    print("Bits recuperados:", bits_recuperados)
    
    # Calcular BER
    ber = sistema.calcular_ber()
    print(f"Tasa de error de bits (BER): {ber:.2f}")
    
    # Graficar señales
    sistema.graficar_señales(t, señal_modulada, señal_ruidosa)
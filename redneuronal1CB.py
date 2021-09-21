{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyN9paAPkBFR/cnLYhWygvOG",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/carlosbrusil/GitCB/blob/master/redneuronal1CB.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "odjWJgjKue-G"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U62XihrKuHFv"
      },
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "f9oIfuyhukF0"
      },
      "source": [
        "celcius=np.array([-40,-10,0,8,15,22,38],dtype=float)\n",
        "farenheit=np.array([-40,14,32,46,59,72,100],dtype=float)"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Cne5LHJ5v2HZ"
      },
      "source": [
        "capa=tf.keras.layers.Dense(units=1,input_shape=[1])     # definimos la capa de salida tipo densa \n",
        "modelo=tf.keras.Sequential([capa])                      # indicamos el modelo secuencial"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "J88SI9P2yJ8k"
      },
      "source": [
        "modelo.compile(\n",
        "  optimizer=tf.keras.optimizers.Adam(0.1),              # Adam 'algoritmo' que ajusta los pesos y sesgos en pasos de 0.1\n",
        "  loss='mean_squared_error'                             # considera el error cuadratico medio 'considra que una poca cantidad de errores grandes es peor que una gran cantidad de errore pequeños'\n",
        ")"
      ],
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hBrfhGjV1KAQ",
        "outputId": "f19fb071-fa8b-48a5-b9aa-4d7637797b93"
      },
      "source": [
        "print (\"comenzando entrenamiento\")\n",
        "historial=modelo.fit(celcius,farenheit,epochs=1000,verbose=False)\n",
        "print (\"entrenamiento terminado\")"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "comenzando entrenamiento\n",
            "entrenamiento terminado\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 283
        },
        "id": "3KIAA8xx2iQs",
        "outputId": "dbd7cad7-3d49-4076-d78e-589c1a7a3c99"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "plt.xlabel=(\"#\")\n",
        "plt.ylabel=(\"Valor Perdida\")\n",
        "plt.plot(historial.history[\"loss\"])"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7feb6fc11a50>]"
            ]
          },
          "metadata": {},
          "execution_count": 13
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX0AAAD4CAYAAAAAczaOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAcpUlEQVR4nO3df3Bd5X3n8ff33qurH5ZlSbYs/AtssMMPb4txFDCbNE2gsQ3N1uxsmpJ2F0/WM/5j6W6605ku7HbG01BmkplOKGy3TLzBrZPJhrAkXXsIG+Ma6KYzxSAXB/wDY2EglvEPYcmWJVu/v/vHea58JUv4ypbulc/5vIY795znPPfc5/gwn/PoOefcY+6OiIgkQ6rUDRARkeJR6IuIJIhCX0QkQRT6IiIJotAXEUmQTKkb8EnmzJnjixcvLnUzRESuKXv27PnY3RvGWjatQ3/x4sU0NzeXuhkiItcUM/twvGUa3hERSRCFvohIgij0RUQSRKEvIpIgCn0RkQRR6IuIJIhCX0QkQWIZ+h+ducB3XjrEkbauUjdFRGRaiWXof9zVy1Mvt3CkrbvUTRERmVZiGfrlmTQAfYNDJW6JiMj0EsvQz2aizeodGCxxS0REppdYhn55LvT71dMXEckX69DX8I6IyEixDP2sevoiImMqKPTNrNbMnjezd8zsoJndbWb1ZrbTzA6H97pQ18zsKTNrMbO3zGxl3nrWh/qHzWz9VG1U7kSuxvRFREYqtKf/JPBzd78FuB04CDwC7HL3ZcCuMA9wH7AsvDYCTwOYWT2wCbgLuBPYlDtQTLaytGEGfQPq6YuI5Lts6JvZLODzwDMA7t7n7meAdcDWUG0r8ECYXgd83yOvAbVmNg9YA+x093Z37wB2AmsndWsutplsOkWvQl9EZIRCevpLgDbgb8zsTTP7npnNABrd/XiocwJoDNMLgKN5n28NZeOVT4nyjEJfRGS0QkI/A6wEnnb3O4BuLg7lAODuDvhkNMjMNppZs5k1t7W1XfF6ysvSCn0RkVEKCf1WoNXdd4f554kOAifDsA3h/VRYfgxYlPf5haFsvPIR3H2zuze5e1NDw5jP9S1INLyjE7kiIvkuG/rufgI4amY3h6J7gQPAdiB3Bc56YFuY3g48FK7iWQWcDcNAO4DVZlYXTuCuDmVTorxMwzsiIqNlCqz3H4EfmlkWOAJ8neiA8ZyZbQA+BL4a6r4I3A+0AOdDXdy93cweA94I9b7p7u2TshVjyKZTunpHRGSUgkLf3fcCTWMsuneMug48PM56tgBbJtLAK6UxfRGRS8XyjlyIrt7p05i+iMgIsQ599fRFREaKd+jrt3dEREaIcein9SubIiKjxDb0sxldpy8iMlpsQ1/DOyIil4p16Gt4R0RkpNiGflY9fRGRS8Q29MszaY3pi4iMEuPQTzHkMKAhHhGRYbEN/eHn5OoGLRGRYbEN/XKFvojIJeIb+mXRw9H1S5siIhfFNvSz6VxPXydzRURyYhv65WUa3hERGS2+oZ/R8I6IyGixDf2LV+9oeEdEJCe2oT989Y7uyhURGRb/0NfNWSIiw2Ib+ln19EVELhHb0M+dyNWYvojIRTEO/WjTdPWOiMhFsQ99XacvInJRQaFvZh+Y2dtmttfMmkNZvZntNLPD4b0ulJuZPWVmLWb2lpmtzFvP+lD/sJmtn5pNilwc3lHoi4jkTKSn/0V3X+HuTWH+EWCXuy8DdoV5gPuAZeG1EXgaooMEsAm4C7gT2JQ7UEyF3B25Gt4REbnoaoZ31gFbw/RW4IG88u975DWg1szmAWuAne7e7u4dwE5g7VV8/yfSb++IiFyq0NB34CUz22NmG0NZo7sfD9MngMYwvQA4mvfZ1lA2XvkIZrbRzJrNrLmtra3A5l0qlTLK0qbhHRGRPJkC633O3Y+Z2Vxgp5m9k7/Q3d3MfDIa5O6bgc0ATU1NV7XO8kxawzsiInkK6um7+7Hwfgr4O6Ix+ZNh2IbwfipUPwYsyvv4wlA2XvmUyWZSGt4REclz2dA3sxlmNjM3DawG9gHbgdwVOOuBbWF6O/BQuIpnFXA2DAPtAFabWV04gbs6lE2Z8kxKd+SKiOQpZHinEfg7M8vV/1/u/nMzewN4zsw2AB8CXw31XwTuB1qA88DXAdy93cweA94I9b7p7u2TtiVjKM+k6NNv74iIDLts6Lv7EeD2McpPA/eOUe7Aw+OsawuwZeLNvDJZ9fRFREaI7R25EJ3I1Zi+iMhFMQ99De+IiOSLdehreEdEZKRYh355JqWbs0RE8sQ89DWmLyKSL9ahX1GWokfDOyIiw2Ie+mku9KunLyKSE/vQ71Hoi4gMi33o6+odEZGLYh760XX6g0OT8gOgIiLXvJiHfvTIRA3xiIhEYh36lQp9EZERYh36FeE5uT26QUtEBIh96KunLyKSL9ahX55R6IuI5It16A8P7yj0RUSAmIf+xRO5GtMXEYGYh77G9EVERkpI6KunLyICsQ/9aPP0o2siIpFYh75uzhIRGSnWoV+u0BcRGaHg0DeztJm9aWYvhPklZrbbzFrM7Mdmlg3l5WG+JSxfnLeOR0P5ITNbM9kbM1pueEePTBQRiUykp/8N4GDe/LeBJ9x9KdABbAjlG4COUP5EqIeZ3QY8CCwH1gJ/bWbpq2v+J8umU5jBhT719EVEoMDQN7OFwG8D3wvzBtwDPB+qbAUeCNPrwjxh+b2h/jrgWXfvdff3gRbgzsnYiE9oNxUZPUhFRCSn0J7+XwJ/AuTGSWYDZ9x9IMy3AgvC9ALgKEBYfjbUHy4f4zPDzGyjmTWbWXNbW9sENmVsldk0PXo4uogIUEDom9mXgVPuvqcI7cHdN7t7k7s3NTQ0XPX6KjJ6OLqISE6mgDqfBX7HzO4HKoAa4Emg1swyoTe/EDgW6h8DFgGtZpYBZgGn88pz8j8zZfScXBGRiy7b03f3R919obsvJjoR+7K7/wHwCvCVUG09sC1Mbw/zhOUvu7uH8gfD1T1LgGXA65O2JeMoV+iLiAwrpKc/nv8CPGtmfw68CTwTyp8BfmBmLUA70YECd99vZs8BB4AB4GF3n/I0rizT8I6ISM6EQt/dXwVeDdNHGOPqG3fvAX53nM8/Djw+0UZeDQ3viIhcFOs7ciGEvq7eEREBEhH6Kd2cJSISJCD00xrTFxEJEhH6vRreEREBkhD6GfX0RURy4h/6ZSk9REVEJEhA6KcZHHL6B9XbFxGJfehXZaNfb1ZvX0QkAaFfmQt9XbYpIhL/0M/19Lt7By5TU0Qk/hIQ+tEvTZxXT19EJAmhrzF9EZGcxIS+hndERBIR+tHwjk7kiogkIvSjnr7G9EVEEhD6lcOhr+EdEZHYh/4MXb0jIjIs9qFfWabhHRGRnNiHfiplVJSlNLwjIkICQh+iIR719EVEEhL6ldm0LtkUESEhoV+VTdOt4R0RkaSEvoZ3RESggNA3swoze93Mfmlm+83sz0L5EjPbbWYtZvZjM8uG8vIw3xKWL85b16Oh/JCZrZmqjRqtSsM7IiJAYT39XuAed78dWAGsNbNVwLeBJ9x9KdABbAj1NwAdofyJUA8zuw14EFgOrAX+2szSk7kx44mGdxT6IiKXDX2PdIXZsvBy4B7g+VC+FXggTK8L84Tl95qZhfJn3b3X3d8HWoA7J2UrLqMqm+GCxvRFRAob0zeztJntBU4BO4H3gDPunkvSVmBBmF4AHAUIy88Cs/PLx/hM/ndtNLNmM2tua2ub+BaNQT19EZFIQaHv7oPuvgJYSNQ7v2WqGuTum929yd2bGhoaJmWdumRTRCQyoat33P0M8ApwN1BrZpmwaCFwLEwfAxYBhOWzgNP55WN8ZkpFN2cN4O7F+DoRkWmrkKt3GsysNkxXAl8CDhKF/1dCtfXAtjC9PcwTlr/sUdpuBx4MV/csAZYBr0/WhnySymyaIYfegaFifJ2IyLSVuXwV5gFbw5U2KeA5d3/BzA4Az5rZnwNvAs+E+s8APzCzFqCd6Iod3H2/mT0HHAAGgIfdvShjLvm/qV9RVpQLhkREpqXLhr67vwXcMUb5Eca4+sbde4DfHWddjwOPT7yZV+fizysPUD8jW+yvFxGZNhJxR26lnp4lIgIkJPT1yEQRkUhCQj8M7/TqBi0RSbZEhP7Miij0uxT6IpJwiQj96nKFvogIJCX01dMXEQGSEvqhp3+uR6EvIsmWiNAvz6QoS5t6+iKSeIkIfTOjujxDl3r6IpJwiQh9iMb1z/X0l7oZIiIllZzQLy/T8I6IJF5iQn9meUYnckUk8ZIT+hUZ9fRFJPESE/rVCn0RkQSFvq7eERFJUOhXZDinnr6IJFxiQn9meYa+gSF6B/TzyiKSXIkJ/dxPMXT3KvRFJLmSE/oVZQAa1xeRREtM6Od+U//sBd2VKyLJlZjQr62MevoKfRFJssSE/qwqhb6IyGVD38wWmdkrZnbAzPab2TdCeb2Z7TSzw+G9LpSbmT1lZi1m9paZrcxb1/pQ/7CZrZ+6zbpUbWUWUOiLSLIV0tMfAP7Y3W8DVgEPm9ltwCPALndfBuwK8wD3AcvCayPwNEQHCWATcBdwJ7Apd6AohllheOfMhb5ifaWIyLRz2dB39+Pu/s9h+hxwEFgArAO2hmpbgQfC9Drg+x55Dag1s3nAGmCnu7e7ewewE1g7qVvzCSrKUmQzKfX0RSTRJjSmb2aLgTuA3UCjux8Pi04AjWF6AXA072OtoWy88tHfsdHMms2sua2tbSLNu1zbmVVZxtnzCn0RSa6CQ9/MqoGfAH/k7p35y9zdAZ+MBrn7ZndvcvemhoaGyVjlsNrKMvX0RSTRCgp9MysjCvwfuvtPQ/HJMGxDeD8Vyo8Bi/I+vjCUjVdeNLVVZZxRT19EEqyQq3cMeAY46O7fyVu0HchdgbMe2JZX/lC4imcVcDYMA+0AVptZXTiBuzqUFc0s9fRFJOEyBdT5LPDvgLfNbG8o+6/At4DnzGwD8CHw1bDsReB+oAU4D3wdwN3bzewx4I1Q75vu3j4pW1GgWZVZDh4/V8yvFBGZVi4b+u7+j4CNs/jeMeo78PA469oCbJlIAyeTevoiknSJuSMXojH9rt4B+geHSt0UEZGSSFzog+7KFZHkSlToz55RDsDpLt2VKyLJlKzQr45+f+fjrt4St0REpDQSFfpzqqOevkJfRJIqYaGf6+lreEdEkilRoT+rsoxMytTTF5HESlTomxmzq7OcVuiLSEIlKvQhGtfX8I6IJFXiQn92dbl6+iKSWIkL/TnVWfX0RSSxEhf6DdXltHX1Ev1EkIhIsiQu9BtrKugbGKJDv6svIgmUuNCfX1sJwEdnLpS4JSIixZe40F8QQv+YQl9EEih5oV8XQr9DoS8iyZO40K+rKqOiLKXhHRFJpMSFvpkxv7aSj84q9EUkeRIX+hCN62t4R0SSKJGhv7CukqMKfRFJoESG/k0N1bR399HerTtzRSRZEhn6S+dWA3D45LkSt0REpLguG/pmtsXMTpnZvryyejPbaWaHw3tdKDcze8rMWszsLTNbmfeZ9aH+YTNbPzWbU5hPNc4E4N1TXaVshohI0RXS0/9bYO2oskeAXe6+DNgV5gHuA5aF10bgaYgOEsAm4C7gTmBT7kBRCvNmVTCzPKOevogkzmVD393/H9A+qngdsDVMbwUeyCv/vkdeA2rNbB6wBtjp7u3u3gHs5NIDSdGYGUsbq3lXoS8iCXOlY/qN7n48TJ8AGsP0AuBoXr3WUDZeecksn1/DvmOdDA7p1zZFJDmu+kSuR79RPGnJaWYbzazZzJrb2toma7WX+Mzierp6Bzh0Qr19EUmOKw39k2HYhvB+KpQfAxbl1VsYysYrv4S7b3b3JndvamhouMLmXV7T4noAXjtyesq+Q0RkurnS0N8O5K7AWQ9syyt/KFzFswo4G4aBdgCrzawunMBdHcpKZkFtJcvmVrPzwMlSNkNEpKgKuWTzR8A/ATebWauZbQC+BXzJzA4DvxXmAV4EjgAtwP8E/gOAu7cDjwFvhNc3Q1lJrV7eyOsftHOys6fUTRERKQqbzo8NbGpq8ubm5ilb/69On+c3/+IVNn7+Rh6979Yp+x4RkWIysz3u3jTWskTekZtz/ewqHlixgC3/+D57j54pdXNERKZcokMf4E9/+1aum1XB7333n3j8ZwfY82E7A4NDpW6WiMiUSPTwTs7Jzh4ee+EAP993goEhp6Yiw28sa+D377qef3nTbMxsytsgIjJZPml4J1PsxkxHjTUV/NXvr+Ts+X5+0dLGL979mJ0HT/Kzt49zy3UzefiLS7n/1+aRTin8ReTapp7+OHr6B9n+y4/47j+8x3tt3dzUMIM/vGcp/+rX55NJJ35UTESmsU/q6Sv0L2NwyPm/+47zVy+38M6Jc9wwu4qHv7CUf71yAWUKfxGZhhT6k2BoyHnpwEn++8uH2f9RJwtqK/m3q27g33x6AXNnVpS6eSIiwxT6k8jdeeXQKZ5+9T3e+KCDdMr44s1z+b3PLOKLNzdo6EdESk4ncieRmXHPLY3cc0sj77V18VzzUX6y5xh/f/AkDTPLeWDFfNatWMDy+TW66kdEph319CdB/+AQrx5q47nmo7x66BT9g87SudXDB4BF9VWlbqKIJIiGd4qoo7uPF/cdZ9ubH/H6B9HPC336hjoeWDGfNcuvY26Nxv9FZGop9EukteM82/Z+xLa9x3j3ZBdmsPL6OtYsb2TN8uu4YfaMUjdRRGJIoV9i7s67J7vYsf8EO/afYP9HnQDcct1MVi+/jtW3NeocgIhMGoX+NHO0/Tw79p/gpf0neePDdtyhYWY5n1/WwBdubuA3ls2htipb6maKyDVKoT+NfdzVy6uH2viHd9v4xeE2zpzvJ2Vw+6JafvNTDXxu6Rx+fWEt2YwuBRWRwij0rxGDQ85brWeGDwK/bD2DO1SUpfj0DXWsWjKbu26cze2LZlGeSZe6uSIyTSn0r1Ed3X3sfr+d146cZvf77bxzohN3KM+kWHl9HStvqGXFojpWLKqlYWZ5qZsrItOEQj8mzpzv4/X323ntSDuvf3Cag8fPMTgU7b+FdZWsWFTLikW13HF9LbfOq6Eqq3vvRJJId+TGRG1VNrraZ/l1AFzoG2TfR2fZ+6sz7D16hn/+sIMX3joOgBksmT2DW+fVcOu8mdw2v4Zb59VwXU2FrhISSTCF/jWsMpvmM4vr+czi+uGyU5097D16hgPHOzl4vJO3j53lZ28fH15eW1XGpxpnclNDNTc1zOCmudXcNKeaBXWVel6ASAIo9GNmbk3FiL8GAM719PPOiXMcDAeCw+GegfbuvuE62UyKG+fM4MaGGdwwewaL6qpYVF/Jwroq5tdW6MSxSEwo9BNgZkXZJX8RALR393GkrYv32rp4r62bI21dHDx+jpf2n2Rg6OK5HjO4rqaChXWVLKqrYn5tJY2zKmicWU5jTQWNNRXMqc7qF0ZFrgEK/QSrn5GlfkY9TaMOBoNDzsnOHo62n+doxwVaO85ztD163/1+Oyc6e4ZPIOekDOZU5w4C5cypLg/rz1JXlaW+Okt9VXa4rCqb1rkFkRIoeuib2VrgSSANfM/dv1XsNsgnS6eM+bWVzK+t5K4xlg8OOae7ejnZ2cvJzh5OdPZwqrOHk529nOjsobXjAr9sPUtHd9+IvxjylWdS1M/IUlNRxsyKTHiVjXivCdM1ldF7ZVmaqmyaymyaqrIMFdkU2XRKBw+RCShq6JtZGvgfwJeAVuANM9vu7geK2Q65OumUMbemgrk1Ffwas8at5+509gzQ0d3H6e4+Orr7aD/fR3uYPt3dR+eFfs71DNDW1cuRj7s51zPAuZ5++gcLu5Q4nTIqy9JU5A4IZdFBIfeeTafIZlKUhfds2qL3EWV583nTmZSRThtpMzIpI5Ua9W5GJiwfqyydGvlKhYOTGaTMMKLnM6QMHbikaIrd078TaHH3IwBm9iywDlDox5CZMauyjFmVZSyeU/gviro7vQNDdPZEB4TcgeFC/yAX+ga50D/I+b5BevoHOd83wIW+IS70D1yy7NS5fvoGhugbGKJ/MFpn/2Bufmjcv0JKKXcAMKIDA9F/0UEib3q4PDXy4AEWDipgYXq0sQ4vhR50Rlcbe/2XFl5pO8Zs1SRv03T1hU818Kdfvm3S11vs0F8AHM2bb4WRIwhmthHYCHD99dcXr2UybZgZFaH3Pnfm1H3P4JBHB4G8A0HuvXdgiMEhv/TlzsCQMzQ06t2dgcFoea5urmwofMYdhsLNkO7OkIM7ONE07jhRnag81M9ND0XLx1wPufUDjPyufGPdiznWoW/sej66oMB1jdGOAr7zatY1duG1ZV5t5ZSsd9qdyHX3zcBmiO7ILXFzJMaiYZfo4CKSFMW+xu4YsChvfmEoExGRIih26L8BLDOzJWaWBR4Ethe5DSIiiVXU4R13HzCzPwR2EF2yucXd9xezDSIiSVb0MX13fxF4sdjfKyIixR/eERGRElLoi4gkiEJfRCRBFPoiIgkyrR+XaGZtwIdXsYo5wMeT1JxrQdK2F7TNSaFtnpgb3L1hrAXTOvSvlpk1j/ecyDhK2vaCtjkptM2TR8M7IiIJotAXEUmQuIf+5lI3oMiStr2gbU4KbfMkifWYvoiIjBT3nr6IiORR6IuIJEgsQ9/M1prZITNrMbNHSt2eyWJmi8zsFTM7YGb7zewbobzezHaa2eHwXhfKzcyeCv8Ob5nZytJuwZUxs7SZvWlmL4T5JWa2O2zXj8PPdGNm5WG+JSxfXMp2Xw0zqzWz583sHTM7aGZ3J2A//+fw//U+M/uRmVXEbV+b2RYzO2Vm+/LKJrxfzWx9qH/YzNZPpA2xC/28h6/fB9wGfM3MJv9Bk6UxAPyxu98GrAIeDtv2CLDL3ZcBu8I8RP8Gy8JrI/B08Zs8Kb4BHMyb/zbwhLsvBTqADaF8A9ARyp8I9a5VTwI/d/dbgNuJtj+2+9nMFgD/CWhy939B9NPrDxK/ff23wNpRZRPar2ZWD2wietTsncCm3IGiIO4eqxdwN7Ajb/5R4NFSt2uKtnUb8CXgEDAvlM0DDoXp7wJfy6s/XO9aeRE9XW0XcA/wAtEzsD8GMqP3N9FzGu4O05lQz0q9DVewzbOA90e3Peb7Off87Pqw714A1sRxXwOLgX1Xul+BrwHfzSsfUe9yr9j19Bn74esLStSWKRP+nL0D2A00uvvxsOgE0Bim4/Bv8ZfAnwBDYX42cMbdB8J8/jYNb29YfjbUv9YsAdqAvwnDWt8zsxnEeD+7+zHgL4BfAceJ9t0e4r+vYeL79ar2dxxDP/bMrBr4CfBH7t6Zv8yjQ38srsM1sy8Dp9x9T6nbUmQZYCXwtLvfAXRz8U9+IF77GSAMT6wjOuDNB2Zw6TBI7BVjv8Yx9GP98HUzKyMK/B+6+09D8UkzmxeWzwNOhfJr/d/is8DvmNkHwLNEQzxPArVmlnvqW/42DW9vWD4LOF3MBk+SVqDV3XeH+eeJDgJx3c8AvwW87+5t7t4P/JRo/8d9X8PE9+tV7e84hn5sH75uZgY8Axx09+/kLdoO5M7gryca68+VPxSuAlgFnM37M3Lac/dH3X2huy8m2o8vu/sfAK8AXwnVRm9v7t/hK6H+NdcbdvcTwFEzuzkU3QscIKb7OfgVsMrMqsL/57ltjvW+Dia6X3cAq82sLvyFtDqUFabUJzWm6ETJ/cC7wHvAfyt1eyZxuz5H9KffW8De8LqfaCxzF3AY+HugPtQ3oiuZ3gPeJroyouTbcYXb/gXghTB9I/A60AL8b6A8lFeE+Zaw/MZSt/sqtncF0Bz29f8B6uK+n4E/A94B9gE/AMrjtq+BHxGds+gn+otuw5XsV+Dfh21vAb4+kTboZxhERBIkjsM7IiIyDoW+iEiCKPRFRBJEoS8ikiAKfRGRBFHoi4gkiEJfRCRB/j81ZPwcmqT4nAAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}
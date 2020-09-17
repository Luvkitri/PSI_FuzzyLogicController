#
# Podstawy Sztucznej Inteligencji, IIS 2020
# Autor: Tomasz Jaworski
# Opis: Szablon kodu do stabilizacji odwróconego wahadła (patyka) w pozycji pionowej podczas ruchu wózka.
#

import gym  # Instalacja: https://github.com/openai/gym
import time
from helper import HumanControl, Keys, CartForce
import matplotlib.pyplot as plt


import numpy as np
import skfuzzy as fuzz

#
# przygotowanie środowiska
#
control = HumanControl()
env = gym.make('gym_PSI:CartPole-v2')
env.reset()
env.render()


def on_key_press(key: int, mod: int):
    global control
    force = 10
    if key == Keys.LEFT:
        control.UserForce = force * CartForce.UNIT_LEFT  # krok w lewo
    if key == Keys.RIGHT:
        control.UserForce = force * CartForce.UNIT_RIGHT  # krok w prawo
    if key == Keys.P:  # pauza
        control.WantPause = True
    if key == Keys.R:  # restart
        control.WantReset = True
    if key == Keys.ESCAPE or key == Keys.Q:  # wyjście
        control.WantExit = True


env.unwrapped.viewer.window.on_key_press = on_key_press

#########################################################
# KOD INICJUJĄCY - do wypełnienia
#########################################################

"""

1. Określ dziedzinę dla każdej zmiennej lingwistycznej. Każda zmienna ma własną dziedzinę.
2. Zdefiniuj funkcje przynależności dla wybranych przez siebie zmiennych lingwistycznych.
3. Wyświetl je, w celach diagnostycznych.

"""
# * POLE ANGLE
# Domain of pole angle
pole_angle_range = np.arange(-180.0, 180.1, 0.1)

# Pole angle membership functions
pole_angle_modifier = 0.1

pole_angle_negative = fuzz.zmf(pole_angle_range, -1.0 * pole_angle_modifier, 0)
pole_angle_zero = fuzz.trimf(pole_angle_range, [-1.0 * pole_angle_modifier, 0, pole_angle_modifier])
pole_angle_positive = fuzz.smf(pole_angle_range, 0, pole_angle_modifier)

# * FORCE
# Domain of force
force_range = np.arange(-10.0, 10.01, 0.01)
force_modifier = 2.0

force_negative = fuzz.zmf(force_range, -1 * force_modifier, 0)
force_zero = fuzz.trimf(force_range, [-1.0 * force_modifier, 0, force_modifier])
force_positive = fuzz.smf(force_range, 0, force_modifier)

if False:
    fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))

    ax0.plot(pole_angle_range, pole_angle_negative, 'b', linewidth=1.5, label='Negative')
    ax0.plot(pole_angle_range, pole_angle_zero, 'g', linewidth=1.5, label='Zero')
    ax0.plot(pole_angle_range, pole_angle_positive, 'r', linewidth=1.5, label='Positive')
    ax0.set_title('Pole Angle')
    ax0.legend()

    plt.tight_layout()
    plt.show()


#########################################################
# KONIEC KODU INICJUJĄCEGO
#########################################################


#
# Główna pętla symulacji
#
while not control.WantExit:

    #
    # Wstrzymywanie symulacji:
    # Pierwsze wciśnięcie klawisza 'p' wstrzymuje; drugie wciśnięcie 'p' wznawia symulację.
    #
    if control.WantPause:
        control.WantPause = False
        while not control.WantPause:
            time.sleep(0.1)
            env.render()
        control.WantPause = False

    #
    # Czy użytkownik chce zresetować symulację?
    if control.WantReset:
        control.WantReset = False
        env.reset()

    ###################################################
    # ALGORYTM REGULACJI - do wypełnienia
    ##################################################

    """
    Opis wektora stanu (env.state)
        cart_position   -   Położenie wózka w osi X. Zakres: -2.5 do 2.5. Ppowyżej tych granic wózka znika z pola widzenia.
        cart_velocity   -   Prędkość wózka. Zakres +- Inf, jednak wartości powyżej +-2.0 generują zbyt szybki ruch.
        pole_angle      -   Pozycja kątowa patyka, a<0 to odchylenie w lewo, a>0 odchylenie w prawo. Pozycja kątowa ma
                            charakter bezwzględny - do pozycji wliczane są obroty patyka.
                            Ze względów intuicyjnych zaleca się konwersję na stopnie (+-180).
        tip_velocity    -   Prędkość wierzchołka patyka. Zakres +- Inf. a<0 to ruch przeciwny do wskazówek zegara,
                            podczas gdy a>0 to ruch zgodny z ruchem wskazówek zegara.
                            
    Opis zadajnika akcji (fuzzy_response):
        Jest to wartość siły przykładana w każdej chwili czasowej symulacji, wyrażona w Newtonach.
        Zakładany krok czasowy symulacji to env.tau (20 ms).
        Przyłożenie i utrzymanie stałej siły do wózka spowoduje, że ten będzie przyspieszał do nieskończoności,
        ruchem jednostajnym.
    """

    cart_position, cart_velocity, pole_angle, tip_velocity = env.state  # Wartości zmierzone

    """
    
    1. Przeprowadź etap rozmywania, w którym dla wartości zmierzonych wyznaczone zostaną ich przynależności do poszczególnych
       zmiennych lingwistycznych. Jedno fizyczne wejście (źródło wartości zmierzonych, np. położenie wózka) posiada własną
       zmienną lingwistyczną.
       
       Sprawdź funkcję interp_membership
       
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.
       Przykład reguły:
       JEŻELI kąt patyka jest zerowy ORAZ prędkość wózka jest zerowa TO moc chwilowa jest zerowa
       JEŻELI kąt patyka jest lekko ujemny ORAZ prędkość wózka jest zerowa TO moc chwilowa jest lekko ujemna
       JEŻELI kąt patyka jest średnio ujemny ORAZ prędkość wózka jest lekko ujemna TO moc chwilowa jest średnio ujemna
       JEŻELI kąt patyka jest szybko rosnący w kierunku ujemnym TO moc chwilowa jest mocno ujemna
       .....
       
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.
    
    
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    
    4. Dla każdej reguły przeprowadź operację wnioskowania Mamdaniego.
       Operatorem wnioskowania jest min().
       Przykład: Jeżeli lingwistyczna zmienna wyjściowa ForceToApply ma 5 wartości (strong left, light left, idle, light right, strong right)
       to liczba wyrażeń wnioskujących wyniesie 5 - po jednym wywołaniu operatora Mamdaniego dla konkluzji.
       
       W ten sposób wyznaczasz aktywacje poszczególnych wartości lingwistycznej zmiennej wyjściowej.
       Uważaj - aktywacja wartości zmiennej lingwistycznej w konkluzji to nie liczba a zbiór rozmyty.
       Ponieważ stosujesz operator min(), to wynikiem będzie "przycięty od góry" zbiór rozmyty. 
       
    5. Agreguj wszystkie aktywacje dla danej zmiennej wyjściowej.
    
    6. Dokonaj defuzyfikacji (np. całkowanie ważone - centroid).
    
    7. Czym będzie wyjściowa wartość skalarna?
    
    """

    # do zmiennej fuzzy_response zapisz wartość siły, jaką chcesz przyłożyć do wózka.
    fuzzy_response = CartForce.IDLE_FORCE

    # * 1. FUZZIFICATION 
    u_pole_angle_negative = fuzz.interp_membership(pole_angle_range, pole_angle_negative, pole_angle)
    u_pole_angle_zero = fuzz.interp_membership(pole_angle_range, pole_angle_zero, pole_angle)
    u_pole_angle_positive = fuzz.interp_membership(pole_angle_range, pole_angle_positive, pole_angle)


    # * 2. RULES
    # IF pole_angle IS negative THEN force is negative
    # IF pole_angle IS positive THEN force is positive

    # There is no logical operators

    # * 3. RULES AGREGATION

    # There is no rule with the same outcome

    # * 4. MAMDANI'S FUZZY INFERENCE METHOD

    u_force_negative = np.fmin(force_negative, u_pole_angle_negative)
    u_force_zero = np.fmin(force_zero, u_pole_angle_zero)
    u_force_positive = np.fmin(force_positive, u_pole_angle_positive)

    # * 5. AGREGATE ALL ACTIVATIONS

    result = np.maximum.reduce([u_force_negative, u_force_zero, u_force_positive])

    # * 6. DEFUZZIFICATION

    fuzzy_response = fuzz.centroid(force_range, result)

    #
    # KONIEC algorytmu regulacji
    #########################

    # Jeżeli użytkownik chce przesunąć wózek, to jego polecenie ma wyższy priorytet
    if control.UserForce is not None:
        applied_force = control.UserForce
        control.UserForce = None
    else:
        applied_force = fuzzy_response

    #
    # Wyświetl stan środowiska oraz wartość odpowiedzi regulatora na ten stan.
    print(
        f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={applied_force:8.4f}")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()

#
# Zostaw ten patyk!
env.close()

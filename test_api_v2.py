# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:22:52 2024

@author: mathi
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from processing import process_data


def check_timestamps_uniqueness(df):
    """Überprüft, ob alle Zeitstempel in einem DataFrame einzigartig sind.
    Es wird angenommen, dass die Zeitstempel in einer Spalte namens 'timestamp' gespeichert sind.

    Args:
        df (pd.DataFrame): Der DataFrame, der die Zeitstempel enthält.

    Returns:
        bool: True, wenn alle Zeitstempel einzigartig sind, sonst False.
    """
    timestamp_column = 'timestamp'  # Der angenommene Name der Zeitstempel-Spalte
    if timestamp_column not in df.columns:
        print(f"Spalte '{timestamp_column}' existiert nicht im DataFrame.")
        return False

    are_unique = df[timestamp_column].is_unique

    return are_unique

def plot_sun_position_reflections_for_multiple_dfs(dataframes, utc):
    # Iteriere durch jeden DataFrame in der Liste
    for i, df in enumerate(dataframes):
        # Konvertierung der Timestamps in das richtige Format und Entfernung von Duplikaten
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Anpassen der Zeit basierend auf dem UTC Offset
        df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=utc)

        # Farbe basierend auf dem aggregierten 'superpositioned_by_sun'
        colors = df['superpositioned_by_sun'].map({True: 'gray', False: 'gold'}).values

        # Erstelle den Plot für den aktuellen DataFrame
        fig, ax = plt.subplots(2, figsize=(12, 16))

        # Scatter Plot
        ax[0].scatter(df['timestamp'], df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0, c=colors, alpha=1, s=10, marker='s')

        # Legende
        legend_labels = {'gold': 'Glare effect', 'gray': 'Reflection is superimposed by the sun'}
        handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=label)
                   for color, label in legend_labels.items()]
        # Anpassung für die Positionierung der Legende unter dem Plot
        if i == len(dataframes) - 1:
            ax[1].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

        # Formatierung der Achsen
        ax[0].xaxis.set_major_locator(mdates.MonthLocator())
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax[0].invert_yaxis()

        # Setze Achsenbeschriftungen
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Time (UTC+' + str(utc) + ')')
        ax[0].set_title(f'Detected glare periods for OP {i+1}')

        # Anpassung der Y-Achse Formatierung
        ax[0].set_ylim(0, 24)
        ax[0].yaxis.set_major_locator(plt.MultipleLocator(1))
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:02d}:00'.format(int(x))))

        # Setze Grenzen für x-Achse auf ein ganzes Jahr
        min_date = df['timestamp'].min().replace(month=1, day=1)
        max_date = df['timestamp'].max().replace(month=12, day=31)
        ax[0].set_xlim(min_date, max_date)

        # Hinzufügen von Gitterlinien für die Stunden im Scatterplot
        ax[0].grid(True, which='both', axis='y')
        ax[0].grid(True, which='major', axis='x')

        # Erstelle das Balkendiagramm für den aktuellen DataFrame
        # Gruppiere Daten nach Datum und 'superpositioned_by_sun' Status
        df_grouped = df.groupby(df['timestamp'].dt.date)['superpositioned_by_sun'].agg(['sum', 'count'])
        df_grouped.rename(columns={'sum': 'true_count', 'count': 'total_count'}, inplace=True)
        df_grouped['false_count'] = df_grouped['total_count'] - df_grouped['true_count']

        # Plot Balkendiagramm
        dates = pd.to_datetime(df_grouped.index)
        ax[1].bar(dates, df_grouped['true_count'], color='gray', label='Reflection is superimposed by the sun', width=1, align='center')
        ax[1].bar(dates, df_grouped['false_count'], bottom=df_grouped['true_count'], color='gold', label='Glare effect', width=1, align='center')

        # Formatierung der X-Achse
        ax[1].xaxis.set_major_locator(mdates.MonthLocator())
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        # Setze Achsenbeschriftungen für Balkendiagramm
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Minutes per day')
        ax[1].set_title(f'Daily glare occurrence for OP {i+1}')

        # Setze Grenzen für x-Achse auf ein ganzes Jahr
        ax[1].set_xlim(min_date, max_date)

        # Hinzufügen von Gitterlinien für die Monate im Balkendiagramm
        ax[1].grid(True, which='major', axis='x')

        plt.tight_layout()

        print("saving data to fig")
        plt. savefig('assets/'+str(utc)+'barchart.png')

        # plt.show()





def runScriptLocally(data):

    pv_areas = data['pv_areas']
    list_of_pv_area_information = data['list_of_pv_area_information']
    list_of_ops = data['list_of_ops']
    utc = data['utc']
    results_as_py_dict =  process_data(pv_areas, list_of_pv_area_information, list_of_ops, utc)

    # results_as_py_dict={'glare_periods':results_as_py_dict}  # [{'timestamp':results_as_py_dict}]}
    #

    dataframes =results_as_py_dict  #  [pd.read_json(json_str) for json_str in results_as_py_dict['glare_periods']]
    calculation_id = data['identifier']
    utc_offset = data['utc']


    if len(dataframes) == 0:
        print("No glare.")
        return False

    else:
        print("Glare derected.")
        print("Are all timestamps unique? " + str(check_timestamps_uniqueness(dataframes[0])))
        plot_sun_position_reflections_for_multiple_dfs(dataframes, utc_offset)
        return True

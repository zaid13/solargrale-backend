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
# from processing import process_data
from new_code.processing_v2 import process_data
import base64
from PIL import Image
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from model.model import Point, GlareRequestModel
import datetime
import pathlib
from pathlib import Path
import os
from dotenv import load_dotenv
from firebase_crud import uploadFileReturnUrl,addUrlTodocument,update_status

import io

from new_code.processing_v6 import calculate_glare


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


def text2png(text, fullpath, color="#000", bgcolor="#FFF", fontfullpath=None, fontsize=15, leftpadding=3,
             rightpadding=3, width=200):
    REPLACEMENT_CHARACTER = u'\uFFFD'
    NEWLINE_REPLACEMENT_STRING = ' ' + REPLACEMENT_CHARACTER + ' '

    # prepare linkback
    # linkback = "created via http://ourdomain.com"
    # fontlinkback = ImageFont.truetype('font.ttf', 8)
    # linkbackx = fontlinkback.getsize(linkback)[0]
    # linkback_height = fontlinkback.getsize(linkback)[1]
    # end of linkback

    font = ImageFont.load_default() if fontfullpath == None else ImageFont.truetype(fontfullpath, fontsize)
    text = text.replace('\n', NEWLINE_REPLACEMENT_STRING)

    lines = []
    line = u""

    for word in text.split():
        print(word)
        if word == REPLACEMENT_CHARACTER:  # give a blank line
            lines.append(line[1:])  # slice the white space in the begining of the line
            line = u""
            lines.append(u"")  # the blank line
        elif font.getsize(line + ' ' + word)[0] <= (width - rightpadding - leftpadding):
            line += ' ' + word
        else:  # start a new line
            lines.append(line[1:])  # slice the white space in the begining of the line
            line = u""

            # TODO: handle too long words at this point
            line += ' ' + word  # for now, assume no word alone can exceed the line width

    if len(line) != 0:
        lines.append(line[1:])  # add the last line

    line_height = font.getsize(text)[1]
    img_height = line_height * (len(lines) + 1)

    img = Image.new("RGBA", (width, img_height), bgcolor)
    draw = ImageDraw.Draw(img)

    y = 0
    for line in lines:
        draw.text((leftpadding, y), line, color, font=font)
        y += line_height

    # add linkback at the bottom
    # draw.text( (width - linkbackx, img_height - linkback_height), linkback, color, font=fontlinkback)

    img.save(fullpath)


def plot_sun_position_reflections_for_multiple_dfs(dataframes, utc, ctr, ops, file_name):
    # Iteriere durch jeden DataFrame in der Liste
    if (len(dataframes) == 0):
        ''''''
        text2png(
            f'No glare detected at Observation point {ctr} \nHeight: {ops["height_above_ground"]} , Latitude:{ops["latitude"]} Longitude:{ops["longitude"]}',
            'assets/' + file_name + f'barchart{ctr}.png', fontfullpath="assets/Lato-Regular.ttf")
        # image = base64.b64decode(str('stringdata'))
        # # fileName = 'assets/'+str(utc)+f'barchart{i+1}.png'

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
        ax[0].scatter(df['timestamp'], df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0, c=colors, alpha=1,
                      s=10, marker='s')

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
        ax[0].set_title(
            f'Detected glare periods for Observation point {ctr} Height: {ops["height_above_ground"]} , Latitude:{ops["latitude"]} Longitude:{ops["longitude"]} ')

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
        ax[1].bar(dates, df_grouped['true_count'], color='gray', label='Reflection is superimposed by the sun', width=1,
                  align='center')
        ax[1].bar(dates, df_grouped['false_count'], bottom=df_grouped['true_count'], color='gold', label='Glare effect',
                  width=1, align='center')

        # Formatierung der X-Achse
        ax[1].xaxis.set_major_locator(mdates.MonthLocator())
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        # Setze Achsenbeschriftungen für Balkendiagramm
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Minutes per day')
        ax[1].set_title(
            f'Daily glare occurrence for Observation point {ctr} Height: {ops["height_above_ground"]} , Latitude:{ops["latitude"]} Longitude:{ops["longitude"]} ')

        # Setze Grenzen für x-Achse auf ein ganzes Jahr
        ax[1].set_xlim(min_date, max_date)

        # Hinzufügen von Gitterlinien für die Monate im Balkendiagramm
        ax[1].grid(True, which='major', axis='x')

        plt.tight_layout()

        print("saving data to fig")
        plt.savefig('assets/' + file_name + f'barchart{ctr}.png')

        # plt.show()


def runScriptLocally(data: GlareRequestModel):
    current_datetime = datetime.datetime.now()
    timestamp = int(current_datetime.timestamp())
    load_dotenv()


    google_api_key =   os.getenv('MAP_KEY')

    pv_areas = data.pv_areas


    list_of_pv_area_information = data.list_of_pv_area_information
    list_of_ops = data.list_of_ops
    meta_data = {'user_id': data.meta_data.user_id, 'project_id': data.meta_data.project_id,
                 'sim_id': data.meta_data.sim_id, 'timestamp': timestamp, 'utc': data.meta_data.utc,"project_name":data.meta_data.project_name}
    simulation_parameter = {'grid_width': data.simulation_parameter.grid_width,
                            'resolution': data.simulation_parameter.resolution,
                            'sun_elevation_threshold': data.simulation_parameter.sun_elevation_threshold,
                            'beam_spread': data.simulation_parameter.beam_spread,
                            'sun_angle': data.simulation_parameter.sun_angle,
                            'sun_reflection_threshold': data.simulation_parameter.sun_reflection_threshold,
                            'zoom_level': data.simulation_parameter.zoom_level}

    pathlib.Path(f'{Path.cwd()}/assets/{str(timestamp)}').mkdir(parents=True, exist_ok=True)
    path=f'{Path.cwd()}/assets/{str(timestamp)}'


    update_status(0.16,data.meta_data.sim_id)
    calculate_glare(pv_areas,
                    list_of_pv_area_information,
                    list_of_ops,
                    meta_data,
                    simulation_parameter,
                    google_api_key,path)


    return str(timestamp)

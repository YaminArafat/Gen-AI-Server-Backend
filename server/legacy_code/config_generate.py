import json
import os
from ollama import chat
from typing import Literal
from pydantic import BaseModel

from background_image_generate import background_image_generator
from typing import Union, List, Optional


class Background(BaseModel):
    color: Literal["green", "red", "black", "blue", "gray", "white", "yellow", "orange", "violet", "purple", "magenta", "pink", "brown", "maroon", "cyan", "null"]
    image: str

class EdgeWidgetItems(BaseModel):
    name: Literal["edge_barometer", "edge_battery", "edge_calendar", "edge_date", "edge_compass",
        "edge_media_controller", "edge_messages", "edge_call_history", "edge_recentApp", "edge_reminder",
        "edge_blood_oxygen", "edge_body_composition", "edge_food", "edge_daily", "edge_hr",
        "edge_sleep", "edge_steps", "edge_stress", "edge_water",
        "edge_women_health", "edge_medication", "edge_circuit_training",
        "edge_cycling", "edge_elliptical", "edge_bike", "edge_hiking",
        "edge_running", "edge_swimming", "edge_treadmill", "edge_walking",
        "edge_weight_machine", "edge_workout", "edge_track_run",
        "edge_blood_pressure", "edge_ecg", "edge_chance_of_rain", "edge_feel_like_temp",
        "edge_finedust", "edge_sunrise", "edge_uv_index", "edge_weather",
        "edge_date_weather", "edge_wind", "edge_alarm", "edge_stopwatch", "edge_timer", "edge_worldclock"]
    type: Literal["edge"]
    position: Literal["bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right", "none"]
    
class CircleWidgetItems(BaseModel):
    name: Literal["circle_compass", "circle_barometer", "circle_battery",
        "circle_calendar_event", "circle_calendar", "circle_media_controller",
        "circle_messages", "circle_moonphase", "circle_call_history",
        "circle_recentApp", "circle_reminder", "circle_altimeter", "circle_ecg",
        "circle_blood_pressure", "circle_blood_oxygen", "circle_body_composition", "circle_food", "circle_daily", "circle_hr", "circle_sleep", "circle_steps", "circle_stress",
        "circle_water", "circle_women_health", "circle_medication", "circle_mood",
        "circle_circuit_training", "circle_cycling", "circle_elliptical", "circle_bike",
        "circle_hiking", "circle_running",
        "circle_swimming", "circle_treadmill", "circle_walking", "circle_weight_machine",
        "circle_workout", "circle_track_run", "circle_alarm",
        "circle_stopwatch", "circle_time", "circle_timer", "circle_worldclock",
        "circle_chance_of_rain", "circle_feel_like_temp", "circle_finedust",
        "circle_sunrise", "circle_uv_index", "circle_weather",
        "circle_wind"]
    type: Literal["circle"]
    position: Literal["top_left", "top", "top_right", "left", "middle", "right", "bottom_left", "bottom", "bottom_right", "none"]
    
class BoxWidgetItem(BaseModel):
    name: Literal["box_calendar", "box_date", "box_reminder",
        "box_hr", "box_steps", "box_stress", "box_blood_oxygen", "box_food", "box_daily", "box_water", "box_medication", "box_air_quality",
        "box_chance_of_rain", "box_detail_weather", "box_temperature", "box_weather", "box_sunrise",
        "box_alarm", "box_timer", "box_worldclock"]
    type: Literal["box"]
    position: Literal["top", "middle", "bottom", "none"]
    
class ArcWidgetItem(BaseModel):
    name: Literal["arc_date", "arc_moonphase", "arc_weekly_steps", "arc_workout_this_week",
        "arc_weekly_activity", "arc_chance_of_rain", "arc_weather",
        "arc_detailed_weather", "arc_temperature"]
    type: Literal["arc"]
    position: Literal["top", "none"]
    
class Widget(BaseModel):
    color: Literal["green", "red", "black", "blue", "gray", "white", "yellow", "orange", "violet", "purple", "magenta", "pink", "brown", "maroon", "cyan"]
    items: List[Union[EdgeWidgetItems, CircleWidgetItems, BoxWidgetItem]]


class Clock(BaseModel):
    type: Literal["digital", "analog"]
    font: Literal[ "simple", "active", "cute", "stamp", "retro_stripe", "bold_serif", "school_time", "cushion", "none"]
    color: Literal["green", "red", "black", "blue", "gray", "white", "yellow", "orange", "violet", "purple", "magenta", "pink", "brown", "maroon", "cyan"]
    position: Literal["top", "middle", "bottom"]

class Dial(BaseModel):
    type: Literal["stick", "roman", "arabic"]
    color: Literal["green", "red", "black", "blue", "gray", "white", "yellow", "orange", "violet", "purple", "magenta", "pink", "brown", "maroon", "cyan"]

class Text(BaseModel):
    content: str
    fontstyle: Literal["italic", "bold", "basic"]
    color: Literal["green", "red", "black", "blue", "gray", "white", "yellow", "orange", "violet", "purple", "magenta", "pink", "brown", "maroon", "cyan"]
    position: Literal["top", "middle", "bottom", "none"]

class ConfigItems(BaseModel):
    Background: Background
    Widget: Widget
    Clock: Clock
    Dial: Dial
    Text: Text
    
class Config(BaseModel):
    Config: ConfigItems
    
def get_Config_config(user_input, host, progress_callback=None):
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_k_m")
    with open('/datasets/prompts.txt', 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    print(user_input)
    input = f"{prompt_text} Now give me a Config configuration return as JSON, no extra JSON. Your input: {user_input}: "

    response = chat(
      messages=[
        {
          'role': 'user',
          'content': input,
        }
      ],
      options={'temperature': 0.5},
      model=model,
      format=Config.model_json_schema(),
    )
    progress_callback(30)
    Config = Config.model_validate_json(response.message.content)
    Config_json = Config.model_dump_json(indent=2)
    Config_data = json.loads(Config_json)
    print(Config_data)
    bg = Config_data["Config"]["Background"]
    color = bg.get("color")
    out_file = ""
    static_path = ""
    if bg.get("image") and bg["image"].startswith("/image/path/to/"):
        placeholder = bg["image"]
        raw_img_prompt = placeholder.split("/")[-1].rsplit(".", 1)[0] #[len("/image/path/to/"):-len(".png")]
        img_prompt = raw_img_prompt.split("ai_generated_", 1)[1] if raw_img_prompt.startswith("ai_generated_") else raw_img_prompt
        print(img_prompt)
        if img_prompt == "default":
            return Config_json, static_path
        extension = placeholder.split('.')[-1]
        progress_callback(40)
        if extension == "gif":
            out_file = f"/server/static/gifs/ai_generated_{img_prompt}.gif"
            local_path = background_image_generator.generate_background_gif("GENERATE_IMAGE good quality gif image for: " + img_prompt, out_file)
            static_path = out_file.split("static/gifs/", 1)[1]
            Config_data["Config"]["Background"]["image"] = f"{host}/static/gifs/{static_path}"
            Config_data["Config"]["Background"]["color"] = "null"
            Config_json = json.dumps(Config_data, indent=2)
        elif extension == "png":
            out_file = f"/server/static/images/ai_generated_{img_prompt}.png"
            local_path = background_image_generator.generate_background_image("GENERATE_IMAGE good quality image for: " + img_prompt, out_file)
            static_path = out_file.split("static/images/", 1)[1]
            Config_data["Config"]["Background"]["image"] = f"{host}/static/images/{static_path}"
            Config_data["Config"]["Background"]["color"] = "null"
            Config_json = json.dumps(Config_data, indent=2)
        print(Config_json)
    progress_callback(90)
    return Config_json, static_path
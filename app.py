from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run
from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import  PersonalityData, PersonalityDataClassifier
from src.pipline.training_pipeline import TrainPipeline
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origin = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origin,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the Personality-Related attributes expected from the form.
    """
    def __init__(self,request:Request):
        self.request: Request = request 
        self.Time_spent_Alone:Optional[float] = None
        self.Stage_fear:Optional[int] = None
        self.Social_event_attendance:Optional[float] = None
        self.Going_outside:Optional[float] = None 
        self.Drained_after_socializing:Optional[int] = None
        self.Friends_circle_size:Optional[float] = None
        self.Post_frequency:Optional[float] = None

    async def get_personality_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.Time_spent_Alone = form.get("Time_spent_Alone")
        self.Stage_fear=form.get("Stage_fear")
        self.Social_event_attendance = form.get("Social_event_attendance")
        self.Going_outside= form.get("Going_outside")
        self.Drained_after_socializing = form.get("Drained_after_socializing")
        self.Friends_circle_size =  form.get("Friends_circle_size")
        self.Post_frequency= form.get("Post_frequency")
        self.Personality = form.get("Personality")
    
# Route to render the main page with the form
@app.get("/",tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for personality data input.
    """
    return templates.TemplateResponse("index.html",{"request":request, "context":"Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training Successful!!!!")
    except Exception as e:
        raise Response(f"Error Occurred!!! {e}")
    
#Routes to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request:Request):
    """
    Endpoint to receive form data, process it, and make prediction.
    """
    try:
        form = DataForm(request)
        await form.get_personality_data()
        personality_data = PersonalityData(
            Time_spent_Alone = form.Time_spent_Alone,
            Stage_fear = form.Stage_fear,
            Social_event_attendance=form.Social_event_attendance,
            Going_outside=form.Going_outside,
            Drained_after_socializing= form.Drained_after_socializing,
            Friends_circle_size= form.Friends_circle_size,
            Post_frequency= form.Post_frequency,
            Personality= form.Personality
            )

        personality_df = personality_data.get_personality_input_frame()

        model_predictor = PersonalityDataClassifier()

        value = model_predictor.predict(personality_df)[0]

        status =  "Personality- Introvert 🙇" if value == 1 else "Personality- Extrovert 🙆"

        return templates.TemplateResponse(
            "index.html",{"request":request ,"context":status})
    
    except Exception as e:
        return {"status":False, "error":f"{e}"}
    
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
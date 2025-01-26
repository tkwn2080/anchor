from typing import Literal, List
from pydantic import BaseModel, Field

# Define the valid options as literals
FreeTimeOption = Literal[
    "Being active and exercising",
    "Reading or learning new things",
    "Spending time outdoors",
    "Creative hobbies",
    "Spending time with friends",
    "Playing video games",
    "Relaxing"
]

CollegePurposeOption = Literal[
    "Pursue new interests",
    "Make new friends",
    "Develop skills for college and career",
    "Have fun and/or relax"
]

SelfDescriptionOption = Literal[
    "Always up for something new",
    "A mix of new and familiar",
    "Prefer sticking to my usual favorites"
]

class UserProfile(BaseModel):
    """User profile with validated categorical responses"""
    free_time: FreeTimeOption = Field(..., description="How free time is spent")
    college_purpose: CollegePurposeOption = Field(..., description="What college should be about")
    happiness_description: str = Field(..., description="Description of happy environments")
    self_description: SelfDescriptionOption = Field(..., description="Self-description category")
    hobbies: str = Field(..., description="List of hobbies and interests")

class SyntheticProfile(BaseModel):
    """Synthetic profile that includes target club information"""
    free_time: FreeTimeOption
    college_purpose: CollegePurposeOption
    happiness_description: str
    self_description: SelfDescriptionOption
    hobbies: str
    target_club_name: str
    target_club_description: str
    secondary_clubs: List[str] = Field(..., description="List of 2 other clubs that would interest this person")

# Export the valid options for use in UI/CLI
VALID_OPTIONS = {
    'free_time': [
        "Being active and exercising",
        "Reading or learning new things",
        "Spending time outdoors",
        "Creative hobbies",
        "Spending time with friends",
        "Playing video games",
        "Relaxing"
    ],
    'college_purpose': [
        "Pursue new interests",
        "Make new friends",
        "Develop skills for college and career",
        "Have fun and/or relax"
    ],
    'self_description': [
        "Always up for something new",
        "A mix of new and familiar",
        "Prefer sticking to my usual favorites"
    ]
} 
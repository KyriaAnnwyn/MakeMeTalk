from biography import Biography
from langchain.prompts import PromptTemplate

from bio_prompts import TOPICS_FROM_BIO_PROMPT
import asyncio
import datetime
import re


# BIO User Simulator
def generate_BIO(user_full_name: str, user_description: str):
    
    bio = Biography(full_name=user_full_name)

    bio.create_lifelong_bio_instructor(user_bio=user_description)
    bio.save_json()

    return bio


def get_relevant_information_from_bio(full_name: str, week_end_date: str = datetime.datetime.now().strftime("%Y-%m-%d")):
    
    # get BIO
    bio = Biography(full_name=full_name)
    
    week_end_date_st = datetime.datetime.strptime(week_end_date, "%Y-%m-%d")
    year_of_birth = bio.get_year_of_birth()
    current_age_num = week_end_date_st.year - year_of_birth + 1
    
    # all days of week
    week_description = "Day 1: "
    num_days = 7
    for day_order_num, days_from_end in enumerate(range(num_days-1, -1, -1)):
        cur_date = week_end_date_st - datetime.timedelta(days=days_from_end)
        print(cur_date)
        current_age_num = cur_date.year - year_of_birth + 1
        week_description += bio.get_existing_day(age_num=current_age_num, mon_order_num=cur_date.month, day_num=cur_date.day)
        if day_order_num < num_days - 1:
            week_description += f"\nDay {day_order_num + 2}: "
    
    # month
    week_start_date_st = week_end_date_st - datetime.timedelta(days=num_days-1)
    month_description = "More precisely, the focus of current month is: \n" + bio.get_existing_month(age_num=current_age_num, mon_order_num=week_end_date_st.month)
    if week_start_date_st.month != week_end_date_st.month:
        if week_start_date_st.year == week_end_date_st.year:
            start_age = current_age_num
        else:
            start_age = week_start_date_st.year - year_of_birth + 1
        month_description += "\n While for the previous month, the focus was the following:\n" + bio.get_existing_month(age_num=start_age, mon_order_num=week_start_date_st.month)
    
    # year
    year_description = "This year you did, are doing and are going to do the following: \n" + bio.get_existing_year(current_age_num)
    if week_start_date_st.year != week_end_date_st.year:
        start_age = week_start_date_st.year - year_of_birth + 1
        year_description += "\n While the previous year you experienced these events:\n" + bio.get_existing_year(start_age)
        
    return year_description, month_description, week_description

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import httpx
from functools import cache, partial

class OpenAiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    OPENAI_API_KEY: SecretStr
    OPENAI_PROXY_SOCKS: str | None = None
    VERBOSE: bool = False
    OPENAI_MODEL_NAME_FALLBACK: str = "gpt-3.5-turbo-1106"
    OPENAI_RESPONSE_TIMEOUT_SECONDS: int = 25

openai_settings = OpenAiSettings()

def get_openai_http_client(is_async=True) -> httpx.AsyncClient | httpx.Client:
    kwargs = {
        "timeout": httpx.Timeout(openai_settings.OPENAI_RESPONSE_TIMEOUT_SECONDS)
    }
    if openai_settings.OPENAI_PROXY_SOCKS:
        kwargs["proxies"] = openai_settings.OPENAI_PROXY_SOCKS
    if is_async:
        http_client = httpx.AsyncClient(**kwargs)
    else:
        http_client = httpx.Client(**kwargs)
    return http_client

@cache
def chat_async_instance(
        model_name: str | None = None, temperature=0.7, max_tokens: int | None = None
) -> ChatOpenAI:
    model_name = model_name if model_name else openai_settings.OPENAI_MODEL_NAME_FALLBACK
    _async_chat = ChatOpenAI(
        api_key=openai_settings.OPENAI_API_KEY.get_secret_value(),
        model=model_name,
        temperature=temperature,
        verbose=openai_settings.VERBOSE,
        max_tokens=max_tokens,
        http_async_client=get_openai_http_client(is_async=True)
    )
    return _async_chat

prompt = PromptTemplate.from_template(TOPICS_FROM_BIO_PROMPT)
model_invoke = chat_async_instance()
topics_chain = prompt | model_invoke

def generate_topics_by_bio(user_full_name : str, week_end_date: str):

    #week description
    # get BIO
    bio = Biography(full_name=user_full_name)

    week_end_date_st = datetime.datetime.strptime(week_end_date, "%Y-%m-%d")
    print(f"week_end_date_st = {week_end_date_st}")
    year_of_birth = bio.get_year_of_birth()
    current_age_num = week_end_date_st.year - year_of_birth + 1

    print(f"current_age_num = {current_age_num}")

    week_description = "Day 1: "
    num_days = 7
    for day_order_num, days_from_end in enumerate(range(num_days-1, -1, -1)):
        cur_date = week_end_date_st - datetime.timedelta(days=days_from_end)
        print(cur_date)
        current_age_num = cur_date.year - year_of_birth + 1
        week_description += bio.get_existing_day(age_num=current_age_num, mon_order_num=cur_date.month, day_num=cur_date.day)
        if day_order_num < num_days - 1:
            week_description += f"\nDay {day_order_num + 2}: "

    print(week_description)

    #month description
    week_start_date_st = week_end_date_st - datetime.timedelta(days=num_days-1)

    month_description = "More precisely, the focus of current month is: \n" + bio.get_existing_month(age_num=current_age_num, mon_order_num=week_end_date_st.month)
    if week_start_date_st.month != week_end_date_st.month:
        if week_start_date_st.year == week_end_date_st.year:
            start_age = current_age_num
        else:
            start_age = week_start_date_st.year - year_of_birth + 1
        month_description += "\n While for the previous month, the focus was the following:\n" + bio.get_existing_month(age_num=start_age, mon_order_num=week_start_date_st.month)
        
    print(month_description) 

    year_description, month_description, week_description = get_relevant_information_from_bio(full_name=user_full_name, week_end_date=week_end_date)

    pattern_to_remove = re.compile("\\n|_")
    context = "\n".join(
        [
            f"{name.capitalize().replace('_', ' ')}: {pattern_to_remove.sub(' ', bio.biography[name])}"
            for name in [
                "full_name",
                "gender",
                "location",
                "category",
                "biography_text",
            ]
        ]
    )

    print(context)

    res = asyncio.run( topics_chain.ainvoke({"bio": bio, "year_description": year_description, "month_description": month_description, "week_description": week_description}))
    
    topics = [re.sub("<topic>|</topic>", "", t).strip() for t in res.content.split('\n')]
    return topics

if __name__ == "__main__":
    user_full_name = "Olivia Silverleaves" 
    user_description = "Olivia Silverleaves is a 26-year-old lifestyle and psychology blogger from Los Angeles, California, captivating an audience primarily of men seeking guidance in their romantic relationships. She started to write blog at 26 years old, 2024, 1 of May.  Embracing her flair for communication and her own tumultuous experiences, Olivia carved out a niche in psychology through self-education and a profound personal journey, rather than traditional academic routes. Her blog began as a personal project, spurred by a deeply transformative episode in her own love life. Olivia's story started with a romance that seemed destined for a fairy tale ending but instead concluded on a somber note. This heartbreak was not the end for Olivia; it was the catalyst for her exploration into the complexities of human emotions and relationships. Through her posts, she combines insightful psychological concepts with practical advice, all while maintaining a charismatic and relatable tone. Olivia's mission is to provide a platform that not only explores the intricacies of male-female dynamics but also offers a beacon of hope and strategies for those navigating the challenging waters of love and relationships. Her approachable style and honest reflections have made her a beloved figure among her followers, who appreciate not just the advice she offers but the genuine care with which she delivers it. Also Olivia is an ordinary attractive lady. She likes fitness, reading psychology books, walking with her friends, parties, shopping and every things an ordinary girls likes. She likes to make provocative photos to attract men to her blog."
    
    #generating initial bio
    bio = generate_BIO(user_full_name = user_full_name, user_description = user_description)

    #Generating topics
    week_end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    topics = generate_topics_by_bio(user_full_name = user_full_name, week_end_date = week_end_date)

    print(topics)

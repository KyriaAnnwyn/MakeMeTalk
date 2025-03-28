import datetime
import json
import re
import os.path
import httpx
from openai import AsyncOpenAI, OpenAI
import instructor
from calendar import monthrange

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from bio_prompts import FORM_BIO_EPOCHS, FORM_YEARS_DESCRIPTION, FORM_MONTHS_DESCRIPTION, FORM_DAY_DESCRIPTION

from pydantic import BaseModel, Field
from typing import Union

from dotenv import load_dotenv
load_dotenv(override=True)

PATH_TO_BIOGRAPHYS = "data/biography.json"

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
PROXY = os.getenv("OPENAI_PROXY_SOCKS")
OPENAI_RESPONSE_TIMEOUT_SECONDS = 25

models = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "gpt-3.5-turbo", "o1"]

_http_client = None
_model_instance = None
_client = None

print(f"Proxy: {PROXY}, openai key {OPEN_AI_API_KEY}")
if PROXY:
    _http_client = httpx.Client(
        proxies=PROXY, timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
    )
else:
    _http_client = httpx.Client(
        timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
    )

_model_instance = OpenAI(
    api_key=OPEN_AI_API_KEY,
    http_client=_http_client
)
_client = instructor.from_openai(_model_instance)

class LifeEpochsDescription(BaseModel):
    epoch_description: list[str]

class BioEpochs(BaseModel):
    year_of_birth: Union[int, None] = Field(
        description="Extract the year of birth of the personage or come up with the appropriate year", default=None
    )
    location: Union[str, None] = Field(
        description="Extract the living location of the personage or come up with the suitable location", default=None
    )
    life_epochs: LifeEpochsDescription = Field(
        description="List of detailed personage life epochs description"
    )
    gender: Union[str, None] = Field(
        description="Correctly assign gender 'Female' or 'Male' of the personage"
    )
    category: Union[str, None] = Field(
        description="Assign the category of interests of the personage"
    )

class EpochYearsDescription(BaseModel):
    year_description: list[str] = Field(min_length=12, max_length=12)

class YearsEpoch(BaseModel):
    epoch_years: EpochYearsDescription = Field(
        description="List of detailed personage year description for an epoch of the life"
    )

class YearMonthsDescription(BaseModel):
    months_description: list[str] = Field(min_length=12, max_length=12)

class MonthsYear(BaseModel):
    months_in_year: YearMonthsDescription = Field(
        description="List of detailed personage months description for a year of the life"
    )

class MonthDaysDescription(BaseModel):
    days_description: list[str] = Field(min_length=28, max_length=31)

class DaysMonth(BaseModel):
    days_in_month: MonthDaysDescription = Field(
        description="List of detailed personage days description for a month of the life"
    )

class Biography(object):

    def __init__(self, full_name: str = None) -> None:
        if full_name is None:
            raise ValueError("full_name must be provided")
        self.all_biographies = []
        if os.path.isfile(PATH_TO_BIOGRAPHYS):
            with open(PATH_TO_BIOGRAPHYS, "r") as f:
                self.all_biographies = json.load(f)
        self.biography = None
        if full_name is not None:
            self.full_name = full_name
            # find out if exists
            for persona in self.all_biographies:
                if persona["full_name"] == full_name:
                    self.biography = persona
                    break
            # new comer
            if not self.biography:
                print("Created new persona")
                self.biography = {
                    "full_name": full_name,
                    "generated_description": {},
                    "facts": [],
                }

    def create_lifelong_bio(self, user_bio: str) -> None:
        self.biography['biography_text'] = user_bio
        #create life apoch description
        print(f"Proxy: {PROXY}, openai key {OPEN_AI_API_KEY}")
        if PROXY:
            http_client = httpx.Client(
                proxies=PROXY, timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
            )
        else:
            http_client = httpx.Client(
                timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
            )

        client = OpenAI(
            api_key=OPEN_AI_API_KEY,
            http_client=http_client
        )

        prompt = FORM_BIO_EPOCHS.replace("{user_bio}", user_bio)

        prompt_message= [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                },
            ]

        params = {
                "model": models[3],
                "messages": prompt_message,
                #"max_tokens": 500,
            }

        result = client.chat.completions.create(**params)
        result = result.choices[0].message.content

        print(f"Generated epochs:\n {result}")

    def create_lifelong_bio_instructor(self, user_bio: str, gender: str) -> None:
        global _client
        self.biography['biography_text'] = user_bio
        #create life epoch description
        if "year_of_birth" in self.biography and "location" in self.biography and "epoch_1" in self.biography:
            return


        prompt = FORM_BIO_EPOCHS.replace("{user_bio}", user_bio)

        resulting_json = _client.chat.completions.create(
                #model=models[3], 
                model="gpt-4o-2024-05-13", 
                response_model=BioEpochs,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                        ]
                    }
                ]
            )
        
        r = json.loads(resulting_json.model_dump_json())
        self.biography["year_of_birth"] = r["year_of_birth"]
        self.biography["location"] = r["location"]
        self.biography["gender"] = r["gender"] if r["gender"] else gender
        self.biography["category"] = r["category"]

        for idx, el in enumerate(r["life_epochs"]["epoch_description"]):
            name_epoch = "epoch_" + str(idx + 1)
            self.biography[name_epoch] = el

    def generate_epoch(self, cur_epoch_num: int) -> None:
        global _client

        epoch_name = "epoch_" + str(cur_epoch_num)
        epoch_description = self.biography[epoch_name]
        prompt = FORM_YEARS_DESCRIPTION.replace("{epoch_description}", epoch_description)

        resulting_json = _client.chat.completions.create(
                #model=models[3], 
                model="gpt-4o-2024-05-13", 
                response_model=YearsEpoch,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                        ]
                    }
                ]
            )
        
        r = json.loads(resulting_json.model_dump_json())

        for idx, el in enumerate(r["epoch_years"]["year_description"]):
            year_num = "year_" + str(idx + 1)
            self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["years"][year_num] = el

    def generate_year(self, cur_epoch_num: int, cur_year_num: int) -> None:
        global _client

        epoch_name = "epoch_" + str(cur_epoch_num)
        year_label = "year_" + str(cur_year_num)
        year_description = self.biography["generated_description"][epoch_name]["years"][year_label] 
        prompt = FORM_MONTHS_DESCRIPTION.replace("{year_description}", year_description)

        resulting_json = _client.chat.completions.create(
                #model=models[3], 
                model="gpt-4o-2024-05-13", 
                response_model=MonthsYear,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                        ]
                    }
                ]
            )
        
        r = json.loads(resulting_json.model_dump_json())

        self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["months"][year_label] = {}
        for idx, el in enumerate(r["months_in_year"]["months_description"]):
            month_num = str(idx + 1).zfill(2) + "_" + datetime.date(1900, idx + 1, 1).strftime("%B").lower() #month_name[idx]
            self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["months"][year_label][month_num] = el

    def generate_month(self, cur_epoch_num: int, cur_year_num: int, mon_nm: str) -> None:
        global _client

        epoch_name = "epoch_" + str(cur_epoch_num)
        year_label = "year_" + str(cur_year_num)
        #year_and_month = str(self.biography["year_of_birth"] + 12*(cur_epoch_num-1) + cur_year_num - 1) + " " + mon_nm[3:]

        number_of_days = monthrange(self.biography["year_of_birth"] + 12*(cur_epoch_num-1) + cur_year_num - 1, int(mon_nm[:2]))
        print(f"Number of days: {number_of_days}")
        month_description = self.biography["generated_description"][epoch_name]["months"][year_label][mon_nm]
        prompt = FORM_DAY_DESCRIPTION.replace("{month_description}", month_description)
        prompt = prompt.replace("{number_of_days}", str(number_of_days[1]))

        resulting_json = _client.chat.completions.create(
                #model=models[3], 
                model="gpt-4o-2024-05-13", 
                response_model=DaysMonth,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                        ]
                    }
                ]
            )
        
        r = json.loads(resulting_json.model_dump_json())

        self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"][year_label] = {}
        self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"][year_label][mon_nm] = {}
        for idx, el in enumerate(r["days_in_month"]["days_description"]):
            #month_num = str(idx + 1).zfill(2) + "_" + datetime.date(1900, idx + 1, 1).strftime("%B").lower() #month_name[idx]
            day_num = str(idx + 1).zfill(2)
            self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"][year_label][mon_nm][day_num] = el

    def add_fact(self, fact_date: str, fact_text: str) -> None:
        """ Adds fact into biography.
        Arguments:
            fact_text (str): text of fact
            fact_date (str): date of fact in format YYYY-MM-DD
        """
        cur_facts = self.biography.get('facts', [])
        cur_facts.append({fact_date: fact_text})
        self.biography['facts'] = cur_facts

    def save_json(self):
        new_persona = True
        for persona in self.all_biographies:
            if persona['full_name'] == self.full_name:
                self.all_biographies.remove(persona)
                self.all_biographies.append(self.biography)
                new_persona = False
                break
        if new_persona:
            self.all_biographies.append(self.biography)
        with open(PATH_TO_BIOGRAPHYS, "w") as f:
            json.dump(self.all_biographies, f, ensure_ascii=False, indent=2)

    def get_existing_biography(self):
        return self.biography["biography_text"]

    def get_year_of_birth(self):
        return self.biography["year_of_birth"]

    def get_existing_epoch(self, num: int):
        return self.biography[f"epoch_{num}"]
    
    def get_user_name(self):
        return self.biography["full_name"]

    @staticmethod
    def year_to_pos(age_num: int):
        prev_epoch_num = age_num // 12
        year_num = age_num % 12
        if year_num != 0:
            cur_epoch_num = prev_epoch_num + 1
            cur_year_num = year_num
        else:
            cur_epoch_num = prev_epoch_num
            cur_year_num = 12
        return cur_epoch_num, cur_year_num

    @staticmethod
    def month_to_name(mon_order_num: int):
        month_full_name = datetime.date(1900, mon_order_num, 1).strftime("%B").lower()
        mon_nm = (
            "0" + str(mon_order_num)
            if len(str(mon_order_num)) == 1
            else str(mon_order_num)
        )
        mon_nm = mon_nm + "_" + month_full_name
        return mon_nm

    def check_existance(self, cur_epoch_num: int | None = None, cur_year_num: int | None = None, age_num: int | None = None, mon_order_num: int | None = None, day_num: int | None = None) -> bool:
        if not self.biography["generated_description"]:
            return False
        if cur_epoch_num and not self.biography["generated_description"][f"epoch_{cur_epoch_num}"]:
            return False
        if cur_year_num and not self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["years"][f"year_{cur_year_num}"]:
            return False
        
        if age_num:
            cur_epoch_num, cur_year_num = self.year_to_pos(age_num)
            if mon_order_num:
                mon_nm = self.month_to_name(mon_order_num)
                if not self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["months"][f"year_{cur_year_num}"][mon_nm]:
                    return False
                if day_num:
                    day_nm = "0" + str(day_num) if len(str(day_num)) == 1 else str(day_num)
                    if not self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"][f"year_{cur_year_num}"][mon_nm][day_nm]:
                        return False
        return True
    
    def generate_on_demand(self, cur_epoch_num: int | None = None, cur_year_num: int | None = None, age_num: int | None = None, mon_order_num: int | None = None, day_num: int | None = None):
        if age_num:
            cur_epoch_num, cur_year_num = self.year_to_pos(age_num)

        if cur_epoch_num and not f"epoch_{cur_epoch_num}" in self.biography["generated_description"]:
            #self.biography["generated_description"][f"epoch_{cur_epoch_num}"] = self.generate_epoch(cur_epoch_num)
            self.biography["generated_description"][f"epoch_{cur_epoch_num}"] = {}
            self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["years"] = {}
            self.generate_epoch(cur_epoch_num)
            
            self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["months"] = {}
            self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"] = {}
            self.save_json()
        if cur_year_num and not f"year_{cur_year_num}" in self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["years"]:
            #self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["years"][f"year_{cur_year_num}"] = 
            self.generate_year(cur_epoch_num, cur_year_num)
            self.save_json()
        
        if age_num:
            if mon_order_num:
                mon_nm = self.month_to_name(mon_order_num)
                if f"year_{cur_year_num}" not in self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["months"]:
                    self.generate_year(cur_epoch_num, cur_year_num)
                    #self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["months"][f"year_{cur_year_num}"][mon_nm] = ""
                    self.save_json()
                if day_num:
                    day_nm = "0" + str(day_num) if len(str(day_num)) == 1 else str(day_num)
                    if f"year_{cur_year_num}" not in self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"] or \
                    mon_nm not in self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"][f"year_{cur_year_num}"] or \
                    day_nm not in self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"][f"year_{cur_year_num}"][mon_nm]:
                        self.generate_month(cur_epoch_num, cur_year_num, mon_nm)
                        #self.biography["generated_description"][f"epoch_{cur_epoch_num}"]["days"][f"year_{cur_year_num}"][mon_nm][day_nm] = ""
                        self.save_json()


    def get_existing_year(self, num: int):
        cur_epoch_num, cur_year_num = self.year_to_pos(num)
        #check if exists
        self.generate_on_demand(cur_epoch_num=cur_epoch_num, cur_year_num=cur_year_num)
        return self.biography["generated_description"][f"epoch_{cur_epoch_num}"][
            "years"
        ][f"year_{cur_year_num}"]

    def get_existing_month(self, age_num, mon_order_num):
        cur_epoch_num, cur_year_num = self.year_to_pos(age_num)
        mon_nm = self.month_to_name(mon_order_num)
        self.generate_on_demand(cur_epoch_num=cur_epoch_num, cur_year_num=cur_year_num, age_num=age_num, mon_order_num=mon_order_num)
        return self.biography["generated_description"][f"epoch_{cur_epoch_num}"][
            "months"
        ][f"year_{cur_year_num}"][mon_nm]

    def get_existing_day(self, age_num, mon_order_num, day_num):
        cur_epoch_num, cur_year_num = self.year_to_pos(age_num)
        mon_nm = self.month_to_name(mon_order_num)
        day_nm = "0" + str(day_num) if len(str(day_num)) == 1 else str(day_num)
        self.generate_on_demand(cur_epoch_num, cur_year_num, age_num, mon_order_num, day_num)
        return self.biography["generated_description"][f"epoch_{cur_epoch_num}"][
            "days"
        ][f"year_{cur_year_num}"][mon_nm][day_nm]


def extract_all_facts_from_chat_history(
        chat_history,
) -> str:
    """This function extracts facts about user from conversation."""
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", max_tokens=2000)
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_prompt_text = """
    You are helpful assistant responsible for keeping the memories of people in order. Today is {today_date}.
    Given chat history extract all facts about the user. Each fact is a description of event happened to user which could be stored as memory of his life. One fact should be inside <fact> and </fact> with its date inside of it between <date> and </date>.
    Follow the next structure:
    <fact> fact №1 description  <date> date of fact №1  </date> </fact>
    <fact> fact №2 description  <date> date of fact №2  </date> </fact>
    ...
    <fact> fact №n description  <date> date of fact №n  </date> </fact>
    
    Write facts in first-person style, do not include hashtags or emoji, create 1-3 sentences description.
    Important: If there are no facts, output phrase `NO NEW FACTS`. For example, if chat history contains only greetings or feelings ("Hello! How are you?", "feel good") and no events related to the user, output `NO NEW FACTS`.
    ### Examples:
    Yesterday I had a challenging task at work related to Langgraph -> <fact> I had a challenging task at work related to Langgraph  <date> 2024-02-29  </date> </fact>
    Last month I played padel -> <fact> I played padel <date> 2024-01-30  </date> </fact>
    Last month I got married and today I am divorcing -> 
        <fact> I got married <date> 2024-01-30  </date> </fact>
        <fact> I am divorcing <date> 2024-03-01  </date> </fact>
    ###
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    gen_text = chain.invoke({"today_date": today_date, "chat_history": chat_history})
    return gen_text["text"]


def format_facts(facts_text: str) -> list[tuple[str, str]] | None:
    if "NO NEW FACTS" in facts_text:
        return None

    facts_text = facts_text.replace("<facts>", "").replace("</facts>", "")
    facts_to_add = []
    tag_regex = re.compile(r"<[a-z]*>|</[a-z]*>")
    if tag_regex.findall(facts_text)[:4] == [
        "<fact>",
        "<date>",
        "</date>",
        "</fact>",
    ]:
        all_facts = [el for el in re.split("</fact>", facts_text) if el != ""]
        for fact in all_facts:
            try:
                fact_text, fact_date = fact.split("<date>")
                fact_date = fact_date.replace("</date>", "").strip()
                try:
                    datetime.datetime.strptime(fact_date, "%Y-%m-%d")
                    facts_to_add.append(
                        (fact_date, fact_text.replace("<fact>", "").strip())
                    )
                except:
                    print("not valid date")
            except:
                print("not valid format for biography")
    elif tag_regex.findall(facts_text)[:4] == [
        "<fact>",
        "</fact>",
        "<date>",
        "</date>",
    ]:
        all_facts = [el for el in re.split("</date>", facts_text) if el != ""]
        for fact in all_facts:
            try:
                fact_text, fact_date = fact.split("</fact>")
                fact_date = fact_date.replace("<date>", "").strip()
                try:
                    datetime.datetime.strptime(fact_date, "%Y-%m-%d")
                    facts_to_add.append(
                        (fact_date, fact_text.replace("<fact>", "").strip())
                    )
                except:
                    print("not valid date")
            except:
                print("not valid format for biography")
    else:
        print("not valid format for biography")
    return facts_to_add

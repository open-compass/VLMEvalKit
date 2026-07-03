DOC_QA_JUDGE_PROMPT = """Now your role is a grading teacher. Your task is to review and score student answers based on reference standard answers. You need to notice the following key points:
- First, extract the final answer from the student's solution, then analyze and judge whether the answer is correct.
- Scoring should only refer to the final answer obtained by the student; there is no need to examine whether the intermediate problem-solving steps are correct.
- When analyzing and judging whether the answer is correct, you need to write down the scoring rationale, organize it into clear statements that follow the logical flow. The summary of the scoring rationale should be placed at the end, using the following format: "In summary, the student's answer deserves x points" (where x represents the student's specific score).
- Keep the whole process concise, within 150 words.
- Provide the score based on your analysis and display it in a code block in "JSON" format.
- An item is covered if it is strictly mentioned or unambiguously implied by a semantic equivalence. This includes numerical equivalence (e.g., 10% and 0.1), synonyms (e.g., UK and United Kingdom), and plural/singular forms (e.g., "apple" and "apples"). However, do not accept loosely related concepts.
Your output format is:
[Scoring Rationale]:
[Score]: x points
[JSON]:
{"answer_score": <integer_value>}

Below is the grading rubric:
[Scores]:
The scoring scale consists of 2 levels in total, from highest to lowest: 1 point, 0 points (the minimum is 0 points; if a situation arises where points need to be deducted beyond 0, simply assign 0 points).
[Tier Details]:
1 point: Assign 1 point if the student's final answer matches the standard answer.
If the question has multiple sub-questions, all sub-questions must be answered correctly to assign 1 point.
0 points:
Assign 0 points if the student's final answer does not match the standard answer.

[Example 1]
<Question>: Based on Document 4153823, answer the following question. What is the total revenue for the organization?
<Standard Answer>: 64933961
<Student Answer>: Answer: 64,933,961.

[Scoring Rationale]: The student's final conclusion is that the total revenue is 64,933,961.
The standard answer is 64933961.
The two numerical values are identical.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Based on Document aguidetoindones, answer the following question. What is Indonesia's GDP in billions of dollars?
<Standard Answer>: 868.3
<Student Answer>: Answer: $868.3 billion

[Scoring Rationale]: The student's final answer is "$868.3 billion".
The standard answer is "868.3".
While the student included the unit and currency symbol, the numerical value matches the standard answer exactly. As the question asks for the value in billions of dollars, the student's answer is semantically equivalent to the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 3]
<Question>: Based on Document 4027862, answer the following question. What was the total retail value(in B$ million) of primary crops, livestock and agrifood processing in 2020 according to the target & trajectory data?
<Standard Answer>: 470.85
<Student Answer>: Answer: 470.86

[Scoring Rationale]: The student's final answer is "470.86".
The standard answer is "470.85".
The student's answer does not match the standard answer numerically. Even though the difference is small, the values are not identical or semantically equivalent.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}

[Example 4]
<Question>: Based on Document 21qualitymaker-, answer the following question. When did C/M Application take place?
<Standard Answer>: SEP,10
<Student Answer>: Answer: Sep,10 and Nov,10

[Scoring Rationale]: The student's final answer is "Sep,10 and Nov,10".
The standard answer is "SEP,10".
Although the student included the correct date, they also provided an additional date ("Nov,10") that is not part of the standard answer. Therefore, the student's answer does not match the standard answer.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}

"""

CURRENT_CASE_PROMPT = """[Current Case]
<Question>: {question}
<Standard Answer>: {reference}
<Student Answer>: {prediction}

"""
ASSISTANT_PROMPT = """[Scoring Rationale]:"""


DOC_QA_LIST_JUDGE_PROMPT = """Now your role is a grading teacher. Your task is to review and score student answers for LIST-style questions, where the standard answer is a list of required items. You need to notice the following key points:
- Analyze and match required items against the student's full response
- Scoring should only refer to the student's answer content; there is no need to examine whether the intermediate problem-solving steps are correct.
- The standard answer is a JSON-like list of items with each item as one required element. Determine whether each item is covered by the student's answer content.
- When analyzing and judging whether each item is covered by the student's answer content, you need to write down the scoring rationale, organize it into clear statements that follow the logical flow. The summary of the scoring rationale should be placed at the end, using the following format: "In summary, the student's answer covers x items" (where x represents the student's specific covered item count).
- Keep the whole process concise, within 200 words.
- Provide the covered item count based on your analysis and display it in a code block in "JSON" format.
- An item is covered if it is strictly mentioned or unambiguously implied by a semantic equivalence. This includes numerical equivalence (e.g., 10% and 0.1), synonyms (e.g., UK and United Kingdom), and plural/singular forms (e.g., "apple" and "apples"). However, do not accept loosely related concepts.
Your output format is:
[Count Rationale]:
[Count]: x
[JSON]:
{"count": [[count]]}

[Example 1]
<Question>: Based on Document 4041252, answer the following question. What are the three top finishes for the Foil Racing Model Freestyle Men's event at the Wind and Water National Kitesurfing Championships 2021?
<Standard Answer>: ["Ahmed Talaat", "Ashraf Luxer", "Ahmed Shaker"]
<Student Answer>: First place: Ahmed Talaat; Second place: Ashraf Luxer; Third place: Ahmed Shaker

[Count Rationale]: The standard answer requires three items: "Ahmed Talaat", "Ashraf Luxer", and "Ahmed Shaker". The student's response lists "Ahmed Talaat" as first place, "Ashraf Luxer" as second place, and "Ahmed Shaker" as third place. All three required names are strictly mentioned in the student's answer. In summary, the student's answer covers 3 items.
[Count]: 3
[JSON]:
{"count": [[3]]}

[Example 2]
<Question>: Based on Document digitalmeasurem, answer the following question. In the Slide that mentioned Qualitative vs Quantitative Measurement, what are the colors of the text "Qualitative" and the background color of it? Please list the colors in list with alphabetical order, e.g., ["black","red"]
<Standard Answer>: ["black", "white"]
<Student Answer>: Got it, let's look at the slide about Qualitative vs Quantitative Measurement. The text "Qualitative" is on a black background, and the text color is white. So the colors are black (background) and white (text). Now, list them in alphabetical order: black, white. The answer is ["black", "white"]

[Count Rationale]: The standard answer requires two items: "black" and "white". The student's response explicitly identifies "black" (background) and "white" (text), and concludes with the correct list ["black", "white"]. Both required items are strictly mentioned and match the standard answer. In summary, the student's answer covers 2 items.
[Count]: 2
[JSON]:
{"count": [[2]]}

[Example 3]
<Question>: Based on Document ddoseattle-1506, answer the following question. According to the chart "Levels of Analytics", what are the four business analystics activities?
<Standard Answer>: ["OPTIMISATION", "PREDICTIVE MODELING", "FORECASTING", "STATISTICAL ANALYSIS"]
<Student Answer>: Statistical Analysis, Forecasting, and Predictive Modelling

[Count Rationale]: The standard answer requires four items: "OPTIMISATION", "PREDICTIVE MODELING", "FORECASTING", and "STATISTICAL ANALYSIS". The student's response covers "Statistical Analysis", "Forecasting", and "Predictive Modelling", but does not include "OPTIMISATION". These correspond to three of the four required items. In summary, the student's answer covers 3 items.
[Count]: 3
[JSON]:
{"count": [[3]]}

"""

DOC_QA_LIST_F1_JUDGE_PROMPT = """Now your role is a grading teacher. Your task is to review and score student answers for LIST-style questions, where the standard answer is a list of required items.
- First, extract the specific list of items from the <Student Answer>. Ignore conversational filler (e.g., "The answer is...").
- Then, compare the [Extracted List] against the <Standard Answer> (Ground Truth).
Here are some extra key points:
- The standard answer is a JSON-like list of items with each item as one required element. Determine whether each item is covered by the student's answer list.
- An item is covered if it is strictly mentioned or unambiguously implied by a semantic equivalence. This includes numerical equivalence (e.g., 10% and 0.1), synonyms (e.g., UK and United Kingdom), and plural/singular forms (e.g., "apple" and "apples"). However, do not accept loosely related concepts.
- You need to write down the extraction and comparing rationale, organize it into clear statements that follow the logical flow. The summary of the rationale should be placed at the end, using the following format: "In summary, the student's answer list has X items, covering Y items from the reference list."
- Keep the whole process concise, within 200 words.
- Provide the student's answer item count and covered item count in a code block in "JSON" format.
Your output format is:
[Rationale]:
[JSON]:
{
    "student_answer_count": <integer_value>,
    "covered_count": <integer_value>
}

[Example 1]
<Question>: Based on Document 4041252, answer the following question. What are the three top finishes for the Foil Racing Model Freestyle Men's event at the Wind and Water National Kitesurfing Championships 2021?
<Standard Answer>: ["Ahmed Talaat", "Ashraf Luxer", "Ahmed Shaker"]
<Student Answer>: First place: Ahmed Talaat; Second place: Ashraf Luxer; Third place: Ahmed Shaker

[Rationale]: The standard answer requires three items: "Ahmed Talaat", "Ashraf Luxer", and "Ahmed Shaker".
The student's response lists "Ahmed Talaat" as first place, "Ashraf Luxer" as second place, and "Ahmed Shaker" as third place. All three required names are strictly mentioned in the student's answer.
In summary, the student's answer list has 3 items, covering 3 items from the reference list.
[JSON]:
{
    "student_answer_count": 3,
    "covered_count": 3
}

[Example 2]
<Question>: Based on Document digitalmeasurem, answer the following question. In the Slide that mentioned Qualitative vs Quantitative Measurement, what are the colors of the text "Qualitative" and the background color of it? Please list the colors in list with alphabetical order, e.g., ["black","red"]
<Standard Answer>: ["black", "white"]
<Student Answer>: Got it, let's look at the slide about Qualitative vs Quantitative Measurement. The text "Qualitative" is on a black background, and the text color is white. So the colors are black (background) and white (text). Now, list them in alphabetical order: black, white. The answer is ["black", "white"]

[Rationale]: The standard answer requires two items: "black" and "white".
The student's response explicitly identifies "black" (background) and "white" (text), and concludes with the correct list ["black", "white"]. Both required items are strictly mentioned and match the standard answer.
In summary, the student's answer list has 2 items, covering 2 items from the reference list.
[JSON]:
{
    "student_answer_count": 2,
    "covered_count": 2
}

[Example 3]
<Question>: Based on Document ddoseattle-1506, answer the following question. According to the chart "Levels of Analytics", what are the four business analytics activities?
<Standard Answer>: ["OPTIMISATION", "PREDICTIVE MODELING", "FORECASTING", "STATISTICAL ANALYSIS"]
<Student Answer>: Statistical Analysis, Forecasting, Predictive Parameter

[Rationale]: The reference list contains four required items: "OPTIMISATION", "PREDICTIVE MODELING", "FORECASTING", and "STATISTICAL ANALYSIS".
The student's answer list includes three items: "Statistical Analysis", "Forecasting", and "Predictive Parameter". These correspond directly to two of the required items: "STATISTICAL ANALYSIS" and "FORECASTING". However, "Predictive Parameter" does not match or unambiguously imply "PREDICTIVE MODELING". The student also did not mention "OPTIMISATION". In summary, the student's answer list has 3 items, covering 2 items from the reference list.
[JSON]:
{
    "student_answer_count": 3,
    "covered_count": 2
}

"""

CURRENT_LIST_CASE_PROMPT = """[Current Case]
<Question>: {question}
<Standard Answer>: {reference}
<Student Answer>: {prediction}

"""


person={
    "name": "小明",
    "age" : 25,
    "city":"北京"
}
print(person["name"])
print(person.get("age"))

person["job"]="工程师"
person["age"]=26

print(person)

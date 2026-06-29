from peewee import Model, CharField

from db import db_instance

class User(Model):
    username = CharField(unique=True)
    password = CharField()

    class Meta:
      
        database = db_instance

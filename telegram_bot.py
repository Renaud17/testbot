from telegram import ReplyKeyboardMarkup
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler,
                          ConversationHandler)
            
import logging

#Habilitamos el logueo 
logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

CHOOSING, TYPING_REPLY, TYPING_CHOICE = range(3)

reply_keyboard = [['Nombre', 'Estado de animo'],
                  ['Promesa importante', 'Color favorito'],
                  ['Estado civil', 'Habilidad o pasatiempo...'],
                  ['Completado']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)

reply_keyboard1 = [['Nombre', 'Estado de animo'],
                  ['Promesa importante', 'Color favorito'],
                  ['Estado civil', 'Habilidad o pasatiempo...'],
                  ['Completado']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)

def facts_to_str(user_data):
    facts = list()

    for key, value in user_data.items():
        facts.append('{} - {}'.format(key, value))

    return "\n".join(facts).join(['\n', '\n'])

def start (bit, update):
    update.message.reply_text(
        "¡Hola! Voy a mantener una conversacion contigo."
        "¿Por que no me dices algo sobre ti?",
        reply_markup=markup)
    return CHOOSING


def help(bot, update):
    update.message.reply_text(
        "¡Hola! Dime que es lo que necesitas.",
        reply_markup=markup)
    return CHOOSING


def regular_choice(bot, update, user_data):
    text = update.message.text
    user_data['choice'] = text
    update.message.reply_text(
        'Cual es tu {}? Me gustaria saber!'.format(text.lower()))

    return TYPING_REPLY

def custom_choice(bot, update):
    update.message.reply_text('Muy bien, por favor enviame lo que desees,'
                              'por ejemplo, "Que te gusta hacer en tu tiempo libre"')

    return TYPING_CHOICE

def received_information(bot, update, user_data):
    text = update.message.text
    category = user_data['choice']
    user_data[category] = text
    del user_data['choice']

    update.message.reply_text("¡Perfecto! Esto es lo que nos dijiste:"
                              "{}"
                              "No deberias estar asi busca alguna actividad en donde te puedas distraer."
                              "Puedes decirme mas o cambiar tu opinion sobre lo que quieras.".format(
                                  facts_to_str(user_data)), reply_markup=markup)
    return CHOOSING
    update.message.reply_text1("¡Perfecto! Esto es lo que nos dijiste:"
                              "{}"
                              "No deberias estar asi busca alguna actividad en donde te puedas distraer."
                              "Puedes decirme mas o cambiar tu opinion sobre lo que quieras.".format(
                                  facts_to_str(user_data)), reply_markup=markup)
    return CHOOSING

def done(bot, update, user_data):
    if 'choice' in user_data:
        del user_data['choice']

    update.message.reply_text("Aprendi estos hechos sobre ti:"
                              "{}"
                              "¡Hasta la proxima!".format(facts_to_str(user_data)))
                            
    user_data.clear()
    return ConversationHandler. END

def error(bot, update, error):
    """Log Errors causados por Updates."""
    logger.warning('La actualizacion "%s" provoco el error "%s"', update, error)

def main():
    #Crear el Actualizador y pasalo el token de tu bot.
    updater = Updater("1083595635:AAHNMxJKqJAcu8k8IHtkkLKTD5zqc8k8kNU")

    #Obtener el despachador para registrar los controladores
    dp = updater.dispatcher

    #Agregue manejador de conversacion con los estados GENDER, PHOTO, LOCATION Y BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            CHOOSING: [RegexHandler('^(Nombre|Estado de animo|Promesa importante||Color favorito|Estado civil|Habilidad o pasatiempo...)$',
                                    regular_choice,
                                    pass_user_data=True),
                        ],

            TYPING_CHOICE: [MessageHandler(Filters.text,
                                           regular_choice,
                                           pass_user_data=True),
                            ],
            TYPING_REPLY: [MessageHandler(Filters.text,
                                           received_information,
                                           pass_user_data=True),
                            ],
        },

        fallbacks=[RegexHandler('^Done$', done, pass_user_data=True)]
    )

    dp.add_handler(conv_handler)


#Agregue manejador de conversacion con los estados GENDER, PHOTO, LOCATION Y BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('help', help)],

        states={
            CHOOSING: [RegexHandler('^(Nombre|Edad|Color favorito|Genero|Numero de hermanos)$',
                                    regular_choice,
                                    pass_user_data=True),
                       RegexHandler('^Algunas cosas...$',
                                    custom_choice),
                        ],

            TYPING_CHOICE: [MessageHandler(Filters.text,
                                           regular_choice,
                                           pass_user_data=True),
                            ],
            TYPING_REPLY: [MessageHandler(Filters.text,
                                           received_information,
                                           pass_user_data=True),
                            ],
        },

        fallbacks=[RegexHandler('^Done$', done, pass_user_data=True)]
    )

    dp.add_handler(conv_handler)


    #log all errors
    dp.add_error_handler(error)

    #Empieza el BOT
    updater.start_polling()

    #Ejecute el bot hasta que presione Ctrl-C o el proceso recibe SIGINT,
    # SIGTERM o sigabrt. Esto debe usarse la mayor parte del teimpo, ya que 
    # start_polling() no bloquea y detendra el bot con gracia.
    updater.idle()

if __name__ == '__main__':
    main()

#Aprende y habla con tu BOT 
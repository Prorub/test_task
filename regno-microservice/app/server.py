from flask import Flask, jsonify, abort, make_response, request
from ml.pick_regno import pick_regno
import logging

model_path = 'ml/models/micromodel.cbm'

logging.basicConfig(filename='logs/logs.log', level=logging.DEBUG)

app = Flask(__name__)


def launch_task(regno_recognize,
                afts_regno_ai,
                recognition_accuracy,
                afts_regno_ai_score,
                afts_regno_ai_char_scores,
                afts_regno_ai_length_scores,
                camera_type,
                camera_class,
                time_check,
                direction,
                api,
                ):

    logging.info('making prediction...')
    prediction = pick_regno(regno_recognize,
                            afts_regno_ai,
                            recognition_accuracy,
                            afts_regno_ai_score,
                            afts_regno_ai_char_scores,
                            afts_regno_ai_length_scores,
                            camera_type,
                            camera_class,
                            time_check,
                            direction,
                            model_name=model_path
                            )

    if api == 'v1.0':
        res_dict = {'result': prediction.tolist()}

        logging.info('getting response')
        return res_dict
    else:
        res_dict = {'error': 'API doesnt exist'}

        logging.warning("api doesn't exist")
        return res_dict


@app.errorhandler(404)
def not_found(error):
    logging.warning('page not found')
    return make_response(jsonify({'code': 'PAGE_NOT_FOUND'}), 404)


@app.errorhandler(500)
def server_error(error):
    logging.warning('server error')
    return make_response(jsonify({'code': 'INTERNAL_SERVER_ERROR'}), 500)


@app.route('/regno/api/v1.0/getpred', methods=['GET'])
def get_task():
    logging.info('launching task')
    result = launch_task(request.args.get('regno_recognize'),
                         request.args.get('afts_regno_ai'),
                         request.args.get('recognition_accuracy'),
                         request.args.get('afts_regno_ai_score'),
                         request.args.get('afts_regno_ai_char_scores'),
                         request.args.get('afts_regno_ai_length_scores'),
                         request.args.get('camera_type'),
                         request.args.get('camera_class'),
                         request.args.get('time_check'),
                         request.args.get('direction'),
                         'v1.0')

    return make_response(jsonify(result), 200)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

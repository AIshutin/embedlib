import os
import logging
import pytest
import dashtable

def pytest_addoption(parser):
    parser.addoption("--max_dataset_size", default=int(1e5))
    parser.addoption("--model_name", default="BERTLike")

def pytest_sessionstart(session):
    session.results = dict()

def pytest_generate_tests(metafunc):
    params = ['max_dataset_size', 'model_name']
    for i, param in enumerate(params):
        result = metafunc.config.getoption(param)
        metafunc.parametrize(param, [result])
        logging.info(f"{param} is {result}")

def pytest_sessionfinish(session, exitstatus):
    print('='*35, 'DL report', '='*35)
    print(f"session exitstatus is {exitstatus}")
    if exitstatus != 0:
        return
    print(session.results)
    parsed_results = []
    params_variety = dict()
    table_params = set()
    static_params = dict()
    for el in session.results:
        options = dict()

        for string in el.split('|'):
            key, value = string.split('=')
            options[key] = value
            if key not in params_variety:
                params_variety[key] = [value]
            elif value not in params_variety[key]:
                    params_variety[key].append(value)
                    table_params.add(key)
        parsed_results.append((options, session.results[el]))

    assert(len(table_params) != 0)
    for el in parsed_results:
        for option in el[0]:
            if option not in table_params:
                static_params[option] = el[0][option]
    print(params_variety)
    table_name = static_params['model_name']
    headers = ['criterion']
    spans = []
    cord = 1
    for i, el in enumerate(sorted(params_variety['metric_name']) + ['time', 'memory']): # inference time, max memory
        headers.append(el)
        curr_span = [[0, cord]]
        cord += 1
        for j in range(len(params_variety['batch_size']) - 1):
            headers.append('')
            curr_span.append([0, cord])
            cord += 1
        spans.append(curr_span)

    subheaders = ['batch_size']
    for i in range((len(headers) - 1) // len(params_variety['batch_size'])):
        for el in params_variety['batch_size']:
            subheaders.append(el)
    cells = dict()
    order = ['criterion_func'] + ['batch_size']
    for el in parsed_results:
        mystr = []
        for param in order:
            mystr.append(el[0][param])
        mystr = '|'.join(mystr)
        if mystr in cells:
            logging.warning("Doubled results. Something went wrong")
        else:
            cells[mystr] = el[1]
    cells_structure = []
    columns = sorted(params_variety['metric_name']) + ['time', 'memory']
    for crit in sorted(params_variety['criterion_func']):
        row = [crit]
        for col in columns:
            for bs in sorted(params_variety['batch_size']):
                results = cells[f'{crit}|{bs}']
                if col not in results:
                    row.append("-")
                else:
                    row.append(f"{results[col]:6.4f}")
        cells_structure.append(row)
    print(table_name)
    print(dashtable.data2rst([headers, subheaders] + cells_structure, spans=spans))

import os
import paramiko
import getpass
import matplotlib as plt
import numpy as np
from configparser import ConfigParser

OF_WIKI = 'wiki'
OF_LATEX = 'latex'
OF_CONSOLE = 'console'

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_separators(output_format):
    if output_format == OF_WIKI:
        start_sep = '|| '
        separator = ' || '
        end_sep = ' ||\n'
        highlight = "'''{}'''"
        pct_token = '%'
    elif output_format == OF_LATEX:
        start_sep = ''
        separator = ' & '
        end_sep = ' \\\\ \\hline\n'
        highlight = "\\textbf{{{0}}}"
        pct_token = '\\%'
    elif output_format == OF_CONSOLE:
        start_sep = '| '
        separator = ' | '
        end_sep = ' |\n'
        highlight = '{0}'
        pct_token = '%'
    else:
        raise ValueError('Unsupported table output format: {}' \
                .format(output_format))

    return {'start': start_sep, 'end': end_sep, 'sep': separator,
            'highlight': highlight, 'pct': pct_token}

def highlight_values(values, output_format):
    highlight = get_separators(output_format)['highlight']
    return [highlight.format(val) for val in values]

def get_val_formatter(v_format, output_format):
    if v_format is None or len(v_format) == 0:
        return lambda val: str(val)
    pct_suffix = ''
    fact = 1.
    if v_format[-1] == '%':
        pct_suffix = get_separators(output_format)['pct']
        fact = 100.
        v_format = v_format[:-1]

    precision = -1
    if len(v_format) > 0:
        precision = int(v_format)

    return lambda val: '{0:.{prec}f}{1}'.format(val * fact,
            pct_suffix, prec=precision)

def get_val_format(val_format, output_format):
    if isinstance(val_format, list):
        return [get_val_formatter(v_format, output_format) for v_format in val_format]
    else:
        return get_val_formatter(val_format, output_format)

def format_values(values, val_formatter):
    if isinstance(val_formatter, list):
        assert len(values) == len(val_formatter)
        return [formatter(val) for val, formatter in \
                zip(values, val_formatter)]
    else:
        return [val_formatter(val) for val in values]

def write_table(wiki_file, col_names, values, output_format=OF_WIKI,
        val_format=None):
    wiki_file.write(get_table(col_names, values, output_format,
        val_format))

def get_table(col_names, values, output_format=OF_WIKI,
        val_format=None):
    sep = get_separators(output_format)
    val_formatter = get_val_format(val_format, output_format)

    if output_format == OF_LATEX:
        table = "\\begin{{tabular}}{{{}}}\n\\hline\n".format("|c" * len(col_names) + "|")
    else:
        table = ""

    table += sep['start'] + sep['sep'].join(
        highlight_values(col_names, output_format)) + sep['end']
    for row in values:
        row = format_values(row, val_formatter)
        table += sep['start'] + sep['sep'].join(row) + sep['end']

    if output_format == OF_LATEX:
        table += "\end{tabular}\n"

    return table

def get_listing(lines, numbered=False, level=0):#, output_format=OF_WIKI):
    linestart = " " * level
    lineitem = (" 1. " if numbered else " * ")
    return linestart + lineitem + ("\n" + linestart + lineitem).join(lines) + "\n"

def get_wiki_link(link_name, link_target):
    return '[[' + link_target + '|' + link_name + ']]'

def expand_file(src_file):
    if os.path.isdir(src_file):
        return [os.path.join(src_file, dir_file) for dir_file in os.listdir(src_file)]
    else:
        return [src_file]

def upload_results(src_files, src_files_prefix, svr_project_path, file_extension_filter=None):
    upload_config = lookup_upload_config()

    BASE_DIR = '/home/' + upload_config['user'] + '/public_html/results/'
    NUM_TRIES = 3
    print('Uploading results in {} to {}...'.format(src_files, upload_config['ssh-addr']))

    src_files = [upload_file for src_file in src_files for upload_file in \
            expand_file(src_file) if \
            file_extension_filter is None or \
            upload_file.endswith(file_extension_filter)]

    if src_files_prefix[-1] != '/':
        src_files_prefix += '/'
    for src_file in src_files:
        assert src_file.startswith(src_files_prefix), \
        "File '{}' does not start with prefix '{}'".format(
                src_file, src_files_prefix)
            
    target_dir = BASE_DIR + svr_project_path + '/'

    with paramiko.SSHClient() as ssh:
        ssh.load_host_keys(os.path.expanduser(os.path.join(
            '~', '.ssh', 'known_hosts')))
        # TODO: this is not secure
        #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        for tries in range(NUM_TRIES):
            credentials = get_credentials()
            if credentials is None:
                print('Aborting result upload')
                return
            try:
                ssh.connect(upload_config['ssh-addr'], username=credentials[0], password=credentials[1])
                #ssh.connect(upload_config['ssh-addr'], username=credentials[0])
                break
            except paramiko.AuthenticationException:
                print('Error authenticating to server')
        credentials = None

        print('Uploading {} files to server'.format(len(src_files)))
        with ssh.open_sftp() as sftp:
            for src_file in src_files:
                shared_src_file_suffix = src_file[len(src_files_prefix):]
                target_file = target_dir + shared_src_file_suffix
                print('copying {} to {}'.format(src_file, target_file))
                sftp.put(src_file, target_file)

def lookup_upload_config():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(file_dir, 'upload_config.ini')

    config_parser = ConfigParser()
    config_parser.read(config_file)

    config = {}
    for config_property in ['ssh-addr', 'web-addr', 'user', 'pass']:
        try:
            config[config_property] = config_parser.get('UploadConfig', config_property)
        except:
            pass
    return config

def get_credentials():
    config = lookup_upload_config()
    # for i in range(3):
    #     username = input('Enter username: ')
    #     if username == '':
    #         return None
    #     password = getpass.getpass('Enter password for {}: '.format(username))
        # return (username, password)
    return (config['user'], config['pass'])

def web_attachment(url_postfix, size=1000):
    upload_config = lookup_upload_config()
    base_url = 'https://{}/~{}/results/'.format(
            'anon-addr',
            'anon-user')
    return '{{' + base_url + url_postfix  + '||width=' + str(size) + '}}'

def set_paper_style():
    #plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 26
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 28
    plt.rcParams['ytick.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 26
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 5.0


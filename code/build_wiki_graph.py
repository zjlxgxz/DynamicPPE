# -*- coding: utf-8 -*-
import re
import os
import gzip
import json
import time
import regex  # pip install regex
import mwxml  # pip install mwxml
import subprocess
import multiprocessing
import numpy as np
import networkx as nx
from mw import Timestamp  # pip install mediawiki-utilities
from datetime import datetime
from os.path import exists as os_exists
from os.path import join as os_join
from os.path import basename as os_basename
from bs4 import BeautifulSoup  # pip install bs4
import urllib.parse
import pickle
import shutil
import urllib.request

lang = "enwiki"
dump_date = "20210101"
base_url = "https://dumps.wikimedia.org/"
web_url = f"https://dumps.wikimedia.org/{lang}/{dump_date}/"
dump_tag = f"{lang}-{dump_date}"
r_path = f"/your/path/data/{lang}-{dump_date}"


def get_list_files_from(file_type):
    """ Get all files of pages-meta-history. All file types are 7z."""
    # wget https://dumps.wikimedia.org/enwiki/20210101/dumpstatus.json
    dumpstatus_path = os_join(r_path, 'dumpstatus.json')
    if not os_exists(dumpstatus_path):
        return None
    json_obj = json.load(open(dumpstatus_path))
    if file_type == 'pages-meta-history':
        keyword = 'metahistory7zdump'
        files = sorted(list(json_obj['jobs'][keyword]['files'].keys()))
        # make the files roughly ordered by created date
        ids = [int(_.split('xml-p')[1].split('p')[0])
               for _ in sorted(files)]
        return [os_join(r_path, file_type, _[1])
                for _ in sorted(zip(ids, files))]


def get_file_from_url(f_base_name):
    """It contains all information of downloadable files."""
    f_name = os_join(r_path, f_base_name)
    if not os_exists(f_name):
        url_ = os_join(base_url, lang, dump_date, f_base_name)
        print(f"downloading {url_}")
        with urllib.request.urlopen(url_) as response:
            shutil.copyfileobj(response, open(os_join(r_path, f_name), 'wb'))
    print(f"load {f_name} finished")


def get_all_ns0_titles():
    """ These articles include redirects. """
    articles = dict()
    # wget it from: https://dumps.wikimedia.org/enwiki/20210101/
    # enwiki-20210101-all-titles-in-ns0.gz
    with open(os_join(r_path, "enwiki-20210101-all-titles-in-ns0")) as f:
        for ind, each_line in enumerate(f):
            if ind == 0:
                continue
            articles[each_line[:-1]] = ind
            if ind % 1000000 == 0:
                print(len(articles), ind)
                assert len(articles) == ind
    # total titles: 15,629,139
    print(f"total number of titles: {len(articles)}")
    return articles


def download_stub_meta_history_files(para):
    """ Download a single page-meta-history file from mirror site. """
    args, f_index = para
    data_root = os_join(args.r_path, f"{args.lang}-{args.date}")
    if not os_exists(data_root):
        os.mkdir(data_root)
    data_root = os_join(data_root, f"{args.file_type}")
    if not os_exists(data_root):
        os.mkdir(data_root)
    f_name = f"{args.lang}-{args.date}-{args.file_type}{f_index}.xml.gz"
    if os_exists(os_join(data_root, f_name)):
        print(f" file {f_name} has already downloaded.")
        return True
    start_time = time.time()
    print(f"downloading: {f_name} ")
    urllib.request.urlretrieve(url=os_join(web_url, f_name),
                               filename=os_join(data_root, f_name))
    print(f"finished in {time.time() - start_time}")
    if not args.extract_flag:
        return True
    # too large, do not use this.
    if not os_exists(os_join(data_root, f_name[:-3])):
        with open(os_join(data_root, f_name[:-3]), 'wb') as outfile:
            outfile.write(gzip.open(f_name, 'rb').read())
    return True


def download_page_meta_history_files(f_path):
    """ Download a single stub-meta-history file from mirror site. """
    f_name = os.path.basename(f_path)
    if not os_exists(r_path):
        os.mkdir(r_path)
    data_root = os_join(r_path, f"pages-meta-history")
    if not os_exists(data_root):
        os.mkdir(data_root)
    if os_exists(os_join(data_root, f_name)):
        print(f" file {f_name} has already downloaded.")
    else:
        print(f"downloading: {f_name}")
        urllib.request.urlretrieve(url=os_join(web_url, f_name),
                                   filename=os_join(data_root, f_name))


def open_7z(filename):
    """Check it out at: https://github.com/WikiLinkGraphs/wikidump"""
    seven_z_path = "/your/path/bin/7za"
    inside_filename, _ = os.path.splitext(os.path.basename(filename))
    args = [seven_z_path, 'e', '-so', filename, inside_filename]
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.stdout


def make_title(title_str: str):
    """ https://en.wikipedia.org/wiki/Category:Wikipedia_naming_conventions"""
    # In Wikipedia, the first character is insensitive.
    # But others are case sensitive.
    if title_str is None:
        return title_str
    if len(title_str) >= 1:
        title_str = title_str.replace(' ', '_')
        # always make the first letter capitalized.
        return ''.join([title_str[0].capitalize(), title_str[1:]])
    # all other cases
    return title_str


def remove_comments(source: str) -> str:
    """Remove all the html comments from a string."""
    pattern = re.compile(r'<!--(.*?)-->', re.MULTILINE | re.DOTALL)
    return pattern.sub('', source)


def get_wikilink_re():
    """ This regex is from the following Github:
    https://github.com/WikiLinkGraphs/wikidump
    """
    regex_str = r'''(?P<total>(?P<wikilink>
        \[\[(?P<link>[^\n\|\]\[\<\>\{\}]{0,256})
        (?:\|(?P<anchor>[^\[]*?))?\]\])\w*)\s?'''
    return regex.compile(regex_str, regex.VERBOSE | regex.MULTILINE)


def timestamp_to_index(cur_date):
    z0 = f"2001-01-21T02:12:21Z"
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    xx = datetime.strptime(z0, date_format)
    return (datetime.strptime(cur_date, date_format) - xx).days


def get_hours(cur_time, pre_time):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    cur_time = datetime.strptime(cur_time, date_format)
    pre_time = datetime.strptime(pre_time, date_format)
    seconds = (cur_time - pre_time).total_seconds()
    return seconds // 3600 + 1


def get_all_ns0_articles():
    start_time = time.time()
    ns0_articles, ns0_created_at = dict(), dict()
    f_name = f'{dump_tag}-all-titles-created-at.json'
    with open(os_join(r_path, f_name)) as f:
        for each_line in f:
            article = json.loads(each_line.rstrip())
            namespace, title = article[0], make_title(article[1])
            redirect, date = make_title(article[2]), article[3]
            if namespace != 0:
                continue
            ns0_articles[title] = redirect
            ns0_created_at[title] = date
    print(f"finished load in {time.time() - start_time:.2f} seconds.")
    return ns0_articles, ns0_created_at


def date_to_int(date_: str):
    """convert datetime format to readable integer"""
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    date_ = datetime.strptime(date_, date_format)
    tmp1 = date_.year * 10 ** 10 + date_.month * 10 ** 8 + date_.day * 10 ** 6
    tmp2 = date_.hour * 10 ** 4 + date_.minute * 10 ** 2 + date_.second
    return int(tmp1 + tmp2)


def articles_created_at(file_path):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    results = []
    start_time = time.time()
    print(f'processing: {file_path}')
    for mw_page in mwxml.Dump.from_file(open_7z(file_path)):
        num_revision = 1
        cur_re = next(mw_page)
        # get the earliest edit.
        for revision in mw_page:
            num_revision += 1
            if cur_re.timestamp > revision.timestamp:
                cur_re = revision
        ns, title = mw_page.namespace, mw_page.title
        redirect_title = mw_page.redirect
        created_at = cur_re.timestamp.strftime(date_format)
        results.append([ns, title, redirect_title, created_at, num_revision])
        del mw_page
    print(f'{os_basename(file_path)} finished in '
          f'{time.time() - start_time:.2f} seconds')
    return results


def process_articles_created_at(num_cpus=1):
    start_time = time.time()
    list_files = get_list_files_from(file_type='pages-meta-history')
    pool = multiprocessing.Pool(processes=num_cpus)
    result_pool = pool.map(func=articles_created_at, iterable=list_files)
    pool.close()
    pool.join()
    print(f'finished in {time.time() - start_time:.2f} seconds')
    f_name = f'{dump_tag}-all-titles-created-at.json'
    with open(os_join(r_path, f_name), 'w') as f:
        for articles in result_pool:
            for article in articles:
                json.dump(article, f)
                f.write('\n')
        f.close()
    f_name = f'{dump_tag}-all-titles-ns0-created-at.json'
    with open(os_join(r_path, f_name), 'w') as f:
        for articles in result_pool:
            for article in articles:
                # only keep articles (namespace=0)
                if article[0] == 0:
                    json.dump(article, f)
                    f.write('\n')
        f.close()


def articles_created_sketch(file_path):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    results = []
    start_time = time.time()
    print(f'processing: {file_path}', flush=True)
    base_name = os_basename(file_path).split('pages-meta-')[1].split('.7z')[0]
    for mw_page in mwxml.Dump.from_file(open_7z(file_path)):
        if mw_page.namespace != 0:
            del mw_page
            continue
        redirect_title = mw_page.redirect
        cur_re = next(mw_page)
        l_timestamp = [cur_re.timestamp.strftime(date_format)]
        for revision in mw_page:
            l_timestamp.append(revision.timestamp.strftime(date_format))
        ns, title = mw_page.namespace, mw_page.title
        results.append([ns, mw_page.id, make_title(title),
                        make_title(redirect_title), base_name, l_timestamp])
        del mw_page
    print(f'{os_basename(file_path)} finished in '
          f'{time.time() - start_time:.2f} seconds', flush=True)
    return results


def process_articles_sketch(num_cpus=1):
    list_files = get_list_files_from(file_type='pages-meta-history')
    pool = multiprocessing.Pool(processes=num_cpus)
    result_pool = pool.map(func=articles_created_sketch, iterable=list_files)
    pool.close()
    pool.join()
    f_name = f'{dump_tag}-all-titles-ns0-sketch.json'
    with open(os_join(r_path, f_name), 'w') as f:
        for articles in result_pool:
            for article in articles:
                # only keep articles (namespace=0)
                if article[0] == 0:
                    json.dump(article, f)
                    f.write('\n')
        f.close()


def extract_internal_links_text(text):
    """ Given a revision element, it extract internal links """
    # https://github.com/WikiLinkGraphs/wikidump
    wikilink_re = get_wikilink_re()
    text = remove_comments(text or '')
    list_links = []
    for match in wikilink_re.finditer(text, concurrent=True):
        link = match.group('link') or ''
        link = link.strip()
        # In some case, the link has been redirected to a sub-section.
        # If this is the case, we just remove the sub-section title.
        if '#' in link:
            splitlink = link.split('#', 1)
            link = splitlink[0]
            if not link:
                link = None  # do not consider self-loop
        if link:
            list_links.append(link)
    return [make_title(_) for _ in list_links if _ is not None]


def get_article_internal_links(para):
    f_path, ns0_titles, ns0_created_at = para
    # the keys of ns0_titles are articles the vales are redirects.
    print(f"processing {f_path}", flush=True)
    start_time = time.time()
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    base_name = os_basename(f_path).split('pages-meta-')[1].split('.7z')[0]
    f = open(os_join(r_path, 'internal_links', base_name + '.json'), 'w')
    for mw_page in mwxml.Dump.from_file(open_7z(f_path)):
        # Ignore the following two cases:
        # 1) articles are not in ns0 and 2) articles are redirects.
        if mw_page.namespace != 0 or mw_page.redirect is not None:
            continue
        page_title = make_title(mw_page.title)
        page_created_at = ns0_created_at[page_title]
        list_valid_links = []
        # warning: each revision is not time ordered.
        for revision in mw_page:
            revision_time = revision.timestamp
            wiki_links = extract_internal_links_text(text=revision.text)
            # check all valid titles and transformed into real articles.
            valid_links = set()
            for link in wiki_links:
                # condition 1: to avoid any non-ns0 link.
                if link not in ns0_titles:
                    continue
                # get the original link
                tmp_link = link if ns0_titles[link] is None \
                    else ns0_titles[link]
                # to make sure the link is also in ns0_created_at
                if link not in ns0_created_at \
                        or tmp_link not in ns0_created_at:
                    continue
                # condition 2: to avoid a red link.
                if Timestamp(ns0_created_at[tmp_link]) > revision_time and \
                        Timestamp(ns0_created_at[link]) > revision_time:
                    continue
                # to add the article itself if there is no redirect
                # otherwise add the corresponding redirect.
                valid_link = link if ns0_titles[link] is None \
                    else ns0_titles[link]
                # ignore self-loop edge
                # It is possible that an article contains the link,
                # which is an article but later on, it becomes redirects
                # This redirect point to the article itself.
                # See: https://en.wikipedia.org/w/index.php?title
                # =Autism&oldid=887247
                # From Wikipedia, the free encyclopedia
                #   (Redirected from Early infantile autism)
                if valid_link != page_title:
                    valid_links.add(valid_link)
            user_id, user_text = None, None
            if revision.user is not None:
                user_id = getattr(revision.user, "id", None)
                user_text = getattr(revision.user, "text", None)
            list_valid_links.append(
                [revision_time, user_id, user_text, valid_links])
        num_add, num_del = 0, 0
        pre_real_links = set()
        # each revision is time ordered.
        for (revision_date, user_id, user_text, valid_links) in sorted(
                list_valid_links, key=lambda x: x[0], reverse=False):
            revision_date = revision_date.strftime(date_format)
            edge_events = []
            for link in valid_links.difference(pre_real_links):
                edge_events.append(("add", link))
                num_add += 1
            for link in pre_real_links.difference(valid_links):
                edge_events.append(("del", link))
                num_del += 1
            pre_real_links = valid_links
            json.dump([page_title, page_created_at,
                       revision_date, user_id, user_text, edge_events], f)
            f.write('\n')
        print(f"{page_title} has {num_add} add and {num_del} del.", flush=True)
        del mw_page
    f.close()
    print(f'finish {os_basename(f_path)} in '
          f'{time.time() - start_time:.2f} seconds', flush=True)


def process_get_article_internal_links(
        num_cpus=55, file_type='pages-meta-history'):
    list_files = get_list_files_from(file_type=file_type)
    ns0_titles, ns0_created_at = get_all_ns0_articles()
    para_space = [(os_join(r_path, file_type, _),
                   ns0_titles, ns0_created_at) for _ in list_files]
    pool = multiprocessing.Pool(processes=num_cpus)
    pool.map(func=get_article_internal_links, iterable=para_space)
    pool.close()
    pool.join()


def get_add_only_edges(para):
    f_path = para
    base_name = os_basename(f_path).split('pages-meta-')[1].split('.7z')[0]
    articles = dict()
    with open(os_join(r_path, f'internal_links/{base_name}.json')) as f:
        for each_line in f:
            items = json.loads(each_line.rstrip())
            p_title, _, re_date, _, _, edge_events = items
            if p_title not in articles:
                articles[p_title] = {
                    'created_at': date_to_int(items[1]),
                    'node_date': dict(), 'remain_links': set()}
            re_date = date_to_int(re_date)
            for event in edge_events:
                if event[0] == "add":
                    # to save the first time added timestamp.
                    if event[1] not in articles[p_title]['node_date']:
                        articles[p_title]['node_date'][event[1]] = re_date
                    articles[p_title]['remain_links'].add(event[1])
                else:
                    articles[p_title]['remain_links'].remove(event[1])
    final_events = dict()
    for p_title in articles:
        final_events[p_title] = {'created_at': articles[p_title]['created_at']}
        final_links = dict()
        for _ in articles[p_title]['remain_links']:
            final_links[_] = articles[p_title]['node_date'][_]
        final_events[p_title]['final_links'] = final_links
    final_edges = []
    final_articles = []
    for p_title in final_events:
        for link, link_date in final_events[p_title]['final_links'].items():
            final_edges.append([link_date, p_title, link])
        final_articles.append([final_events[p_title]['created_at'], p_title])
    final_edges = sorted(final_edges, key=lambda x: x[0])
    final_articles = sorted(final_articles, key=lambda x: x[0])
    return final_edges, final_articles


def process_add_only_dynamic_graph(
        num_cpus=60, file_type='pages-meta-history'):
    """
    This is for added-only graph (only include add events).
    :param num_cpus:
    :param file_type:
    :return:
    """
    list_files = get_list_files_from(file_type=file_type)
    para_space = [os_join(r_path, file_type, _) for _ in list_files]
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(func=get_add_only_edges, iterable=para_space)
    pool.close()
    pool.join()
    all_edge_list = []
    all_article_list = []
    for edge_list, article_list in results_pool:
        all_edge_list.extend(edge_list)
        all_article_list.extend(article_list)
    f_write = open(os_join(
        r_path, f"{dump_tag}-add-only-nodes-tmp.json"), 'w')
    for _ in sorted(all_article_list):
        json.dump(_, f_write)
        f_write.write('\n')
    f_write.close()
    f_write = open(os_join(
        r_path, f"{dump_tag}-add-only-edges-tmp.json"), 'w')
    for _ in sorted(all_edge_list):
        json.dump(_, f_write)
        f_write.write('\n')
    f_write.close()


def special_cases(uu):
    # these four are the articles redirected incorrectly.
    if uu == 'Pagliaccio':
        return 'Pagliacci'
    if uu == 'Fronberg':
        return 'Schloss_Fronberg'
    if uu == 'Giuseppe_Riccobaldi':
        return 'Giuseppe_Riccobaldi_del_Bava'
    if uu == 'Automobilka_Gatter':
        return 'Gatter_Autowerk_Reichstadt'


def prepare_add_only_nodes():
    nodes_id_map = dict()
    f_node_path = os_join(r_path, f"{dump_tag}-add-only-nodes-tmp.json")
    f_label_path = os_join(r_path, f"{dump_tag}-all-sorted-labels.json")
    f_write = open(os_join(r_path, f"en-wiki-add-only-nodes.json"), 'w')
    with open(f_node_path, 'rb') as f_node:
        with open(f_label_path, 'rb') as f_label:
            for ind, (l_node, l_label) in enumerate(zip(f_node, f_label)):
                created_at1, title1 = json.loads(l_node.rstrip())
                created_at2, title2, label = json.loads(l_label.rstrip())
                assert created_at1 == created_at2
                assert title1 == title2
                nodes_id_map[title1] = ind
                json.dump([ind, title1, created_at1, label], f_write)
                f_write.write(f"\n")
    f_write.close()


def prepare_add_only_edges():
    nodes_id_map = dict()
    f_node_path = os_join(r_path, f"{dump_tag}-add-only-nodes-tmp.json")
    with open(f_node_path, 'rb') as f_node:
        for ind, l_node in enumerate(f_node):
            created_at1, title1 = json.loads(l_node.rstrip())
            nodes_id_map[title1] = ind
    ns0_titles, ns0_created_at = get_all_ns0_articles()
    f_edge_path = os_join(r_path, f"{dump_tag}-add-only-edges-tmp.json")
    f_write = open(os_join(r_path, f"en-wiki-add-only-edges.json"), 'w')
    with open(f_edge_path, 'rb') as f_edge:
        for l_edge in f_edge:
            timestamp, uu, vv = json.loads(l_edge.rstrip())
            index = 0
            # redirects could be a chain
            while uu not in nodes_id_map:
                uu = ns0_titles[uu]
                index += 1
                if index > 100:
                    break
            if uu not in nodes_id_map:
                print(f"self-loop: {uu}")
                uu = special_cases(uu)
            index = 0
            while vv not in nodes_id_map:
                vv = ns0_titles[vv]
                index += 1
                if index > 100:
                    break
            if vv not in nodes_id_map:
                print(f"self-loop: {vv}")
                vv = special_cases(vv)
            if uu not in nodes_id_map or vv not in nodes_id_map:
                print(f"bad edge: {uu} {vv}")
                continue
            edge_rec = [nodes_id_map[uu], nodes_id_map[vv], timestamp, "Add"]
            json.dump(edge_rec, f_write)
            f_write.write(f"\n")
    f_write.close()


def build_add_only_graph():
    nodes = [[0, 0] for _ in range(6220822)]
    with open(os_join(r_path, f"en-wiki-add-only-edges.json")) as f_edge:
        for ind, each_line in enumerate(f_edge):
            uu, vv, created_at, event = json.loads(each_line)
            if created_at >= 20210101000000:
                continue
            nodes[uu][0] += 1
            nodes[vv][1] += 1
            if ind % 1000000 == 0:
                print(ind)
    re_index = dict()
    index = 0
    fw = open(os_join(r_path, f"en-wiki-nodes-.json"), 'w')
    with open(os_join(r_path, f"en-wiki-add-only-nodes.json")) as f_node:
        for ind, each_line in enumerate(f_node):
            node_id, title, created_at, label = json.loads(each_line)
            # consider articles before 2021
            if created_at >= 20210101000000:
                continue
            # remove isolated articles
            if nodes[node_id][0] == 0 and nodes[node_id][1] == 0:
                continue
            json.dump([index, title, created_at, label], fw)
            fw.write('\n')
            re_index[node_id] = index
            index += 1
    fw.close()
    fw = open(os_join(r_path, f"en-wiki-edges-.json"), 'w')
    new_nodes = dict()
    with open(os_join(r_path, f"en-wiki-add-only-edges.json")) as f_edge:
        for ind, each_line in enumerate(f_edge):
            uu, vv, created_at, event = json.loads(each_line)
            # consider articles before 2021
            if created_at >= 20210101000000:
                continue
            if uu not in re_index or vv not in re_index:
                print(uu, vv)
                continue
            json.dump([re_index[uu], re_index[vv], created_at, event], fw)
            fw.write('\n')
            new_nodes[re_index[uu]] = ''
            new_nodes[re_index[vv]] = ''
    fw.close()
    print(len(new_nodes), index)
    nodes = dict()
    with open(os_join(r_path, f"en-wiki-nodes-.json")) as f_node:
        for ind, each_line in enumerate(f_node):
            node_id, title, created_at, label = json.loads(each_line)
            nodes[node_id] = ''
    g = nx.Graph()
    with open(os_join(r_path, f"en-wiki-edges-.json")) as f_node:
        for ind, each_line in enumerate(f_node):
            uu, vv, timestamp, event = json.loads(each_line)
            g.add_edge(uu, vv)
            if ind % 1000000 == 0:
                print(ind)
    print(len(nodes), len(g.nodes))
    for ind, cc in enumerate(nx.connected_components(g)):
        print(ind, len(cc))


def get_events_gaps(para):
    f_path = para
    base_name = os_basename(f_path).split('pages-meta-')[1].split('.7z')[0]
    articles = dict()
    with open(os_join(r_path, f'enwiki-links/{base_name}.json')) as f:
        for ind, each_line in enumerate(f):
            items = json.loads(each_line.rstrip())
            article, create_time, re_time, _, _, link_events = items
            if article not in articles:
                articles[article] = []
            articles[article].append((re_time, len(link_events)))
    revision_gaps = []
    event_gaps = []
    for article in articles:
        pre_re_time = None
        for item in articles[article]:
            cur_re_time = item[0]
            if pre_re_time is not None:
                revision_gaps.append(get_hours(cur_re_time, pre_re_time))
            pre_re_time = cur_re_time
        pre_event_time = None
        for item in [_ for _ in articles[article] if _[1] != 0]:
            cur_event_time = item[0]
            if pre_event_time is not None:
                event_gaps.append(get_hours(cur_event_time, pre_event_time))
            pre_event_time = cur_event_time
    return revision_gaps, event_gaps


def process_event_gaps(
        num_cpus=60, file_type='pages-meta-history'):
    list_files = get_list_files_from(file_type=file_type)
    para_space = [os_join(r_path, file_type, _) for _ in list_files]
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(func=get_events_gaps, iterable=para_space)
    pool.close()
    pool.join()
    f_write = open(os_join(r_path, f"results-stats-gaps.json"), 'w')
    for revision_gap, event_gap in results_pool:
        json.dump([revision_gap, event_gap], f_write)
        f_write.write(f"\n")
    f_write.close()


def plot_event_gaps():
    import matplotlib.pyplot as plt
    groups_re_gap = dict()
    groups_event_gap = dict()
    fr = open(os_join(r_path, f"results-stats-gaps.json"), 'rb')
    for each_line in fr:
        revision_gap, event_gap = json.loads(each_line.rstrip())
        print(len(revision_gap), len(event_gap))
        for _ in revision_gap:
            if _ not in groups_re_gap:
                groups_re_gap[_] = 0
            groups_re_gap[_] += 1
        for _ in event_gap:
            if _ not in groups_event_gap:
                groups_event_gap[_] = 0
            groups_event_gap[_] += 1
    pickle.dump([groups_re_gap, groups_event_gap],
                open(os_join(r_path, f"results-stats-gaps-groups.json"), 'wb'))
    x, y = [], []
    for key, val in sorted(
            groups_re_gap.items(), key=lambda _: _[1], reverse=True):
        x.append(key)
        y.append(val)
    plt.loglog(x, y, 'ro', label='Revision', markerfacecolor=None)
    x, y = [], []
    for key, val in sorted(
            groups_event_gap.items(), key=lambda _: _[1], reverse=True):
        x.append(key)
        y.append(val)
    plt.loglog(x, y, 'b+', label="Event")
    plt.ylabel('Number of edits')
    plt.xlabel('Edit gap (hours)')
    plt.legend()
    plt.savefig(r_path + "/figs/re-event-gaps.pdf",
                dpi=300, bbox_inches='tight', pad_inches=.02,
                format='pdf')
    plt.close()


def merge_events(f_path, threshold):
    def _merge_revisions(list_revisions):
        added = set()
        deleted = set()
        for revision in list_revisions:
            for _ in revision[-1]:
                if _[0] == 'add':
                    added.add(_[1])
                if _[0] == 'del':
                    deleted.add(_[1])
        iter_sect = added.symmetric_difference(deleted)
        re_events = [("add", _) for _ in added if _ not in iter_sect]
        re_events.extend([("del", _) for _ in deleted if _ not in iter_sect])
        list_revisions[-1][-1] = re_events
        return list_revisions[-1]

    pages = dict()
    pages_events = dict()
    all_edge_events = []
    num_all_added = 0
    num_all_deleted = 0
    all_added_articles = dict()
    all_deleted_articles = dict()
    with open(f_path) as f:
        for ind, each_line in enumerate(f):
            items = json.loads(each_line.rstrip())
            if items[0] not in pages:
                pages[items[0]] = []
                pages_events[items[0]] = []
            pages[items[0]].append(items)
            add_events = dict()
            del_events = dict()
            tmp_events = []
            for event_type, to_article in items[-1]:
                if event_type == 'add':
                    edge_event = [items[0], to_article, items[2], "Add"]
                    all_added_articles[to_article] = ''
                    add_events[to_article] = ''
                    num_all_added += 1
                else:
                    edge_event = [items[0], to_article, items[2], "Del"]
                    all_deleted_articles[to_article] = ''
                    del_events[to_article] = ''
                    num_all_deleted += 1
                tmp_events.append(edge_event)
            inter_sect_events = set(add_events.keys()) \
                .symmetric_difference(set(del_events.keys()))
            for _ in tmp_events:
                if _[1] not in inter_sect_events:
                    pages_events[items[0]].append(_)
                    all_edge_events.append(_)
    for title in pages:
        list_items = pages[title]
        pre_date_int = date_to_int(list_items[0][2])
        merged_items = [[list_items[0]]]
        # merged_items[-1] = _merge_revisions([merged_items[-1], items])
        for items in list_items[1:]:
            page_title, _, revision_date, _, _, edge_events = items
            date_int = date_to_int(revision_date)
            if date_int - pre_date_int <= threshold:
                merged_items[-1].append(items)
            else:
                merged_items.append([items])
            pre_date_int = date_int
        for index, items in enumerate(merged_items):
            if len(items) == 1:
                merged_items[index] = items[0]
            else:
                merged_items[index] = _merge_revisions(items)
        pages[title] = merged_items
    edge_events = []
    num_reduced_added = 0
    num_reduced_deleted = 0
    red_added_articles = dict()
    red_deleted_articles = dict()
    for article in pages:
        for item in pages[article]:
            add_events = dict()
            del_events = dict()
            tmp_events = []
            for event_type, to_article in item[-1]:
                if event_type == 'add':
                    edge_event = [item[0], to_article, item[2], "Add"]
                    red_added_articles[to_article] = ''
                    add_events[to_article] = ''
                    num_reduced_added += 1
                else:
                    edge_event = [item[0], to_article, item[2], "Del"]
                    red_deleted_articles[to_article] = ''
                    del_events[to_article] = ''
                    num_reduced_deleted += 1
                tmp_events.append(edge_event)
            inter_sect_events = set(add_events.keys()) \
                .symmetric_difference(set(del_events.keys()))
            for _ in tmp_events:
                if _[1] not in inter_sect_events:
                    edge_events.append(_)
    print('---')
    print(f"all edge events: {len(all_edge_events)}, "
          f"added: {num_all_added} deleted: {num_all_deleted}")
    print(f"num-add-articles: {len(all_added_articles)} "
          f"num-deleted-articles: {len(all_deleted_articles)}")
    print(f"reduced: {len(edge_events)} "
          f"added: {num_reduced_added} deleted: {num_reduced_deleted}")
    print(f"num-add-articles: {len(red_added_articles)} "
          f"num-deleted-articles: {len(red_deleted_articles)}"
          f" remained: {len(red_added_articles) - len(red_deleted_articles)}")
    print('---')
    return all_edge_events, edge_events


def draw_article_events(page_data):
    import matplotlib.pyplot as plt
    index = 0
    created_at = page_data['created-at']
    latest_date = '2020-09-01T23:59:59Z'
    to_article_events = dict()
    for re_date, events in page_data['list-events']:
        for event_type, to_article in events:
            if to_article not in to_article_events:
                to_article_events[to_article] = []
            hours = get_hours(re_date, created_at) // 24 + 1
            to_article_events[to_article].append(hours)
    for ind, to_page in enumerate(to_article_events):
        if len(to_article_events[to_page]) % 2 == 1:
            hours = get_hours(latest_date, created_at) // 24 + 1
            to_article_events[to_page].append(hours)
            for i in range(len(to_article_events[to_page]) // 2):
                plt.plot(
                    to_article_events[to_page][2 * i:2 * i + 2],
                    [index] * 2, c='r', linewidth=.2, marker='+', markersize=.8)
            index += 1
    for ind, to_page in enumerate(to_article_events):
        if len(to_article_events[to_page]) % 2 == 0:
            plt.plot(to_article_events[to_page],
                     [index] * len(to_article_events[to_page]),
                     c='b', linewidth=.2, marker='o', markersize=.8)
            index += 1
    plt.show()


def get_all_edges(para):
    f_path = para
    pages = dict()
    for ind, each_line in enumerate(open(f_path)):
        items = json.loads(each_line.rstrip())
        title, created_at, list_events = items[0], items[1], items[5]
        revision_time = date_to_int(items[2])
        if title not in pages:
            pages[title] = dict()
        for event_type, to_article in list_events:
            if to_article not in pages[title]:
                pages[title][to_article] = []
            if event_type == 'add':
                edge_event = [revision_time, title, to_article, "Add"]
            else:
                edge_event = [revision_time, title, to_article, "Del"]
            pages[title][to_article].append(edge_event)
    list_edge_events = []
    for page in pages:
        for to_article in pages[page]:
            if pages[page][to_article][-1][-1] == "Del":
                assert len(pages[page][to_article]) % 2 == 0
                aa = pages[page][to_article][0]
                bb = pages[page][to_article][-1]
                list_edge_events.append(aa)
                list_edge_events.append(bb)
            else:
                aa = pages[page][to_article][0]
                list_edge_events.append(aa)
    print(len(list_edge_events))
    return list_edge_events


def count_add_del_events(para):
    f_path = para
    articles = dict()
    for ind, each_line in enumerate(open(f_path)):
        items = json.loads(each_line.rstrip())
        title, created_at, list_events = items[0], items[1], items[5]
        if title not in articles:
            articles[title] = {'num-add': 0, 'num-del': 0, 'num-revisions': 0,
                               'num-nontrivial-revisions': 0,
                               'add-articles': set(), 'del-articles': set()}
        articles[title]['num-revisions'] += 1
        if len(list_events) != 0:
            articles[title]['num-nontrivial-revisions'] += 1
            for event_type, to_article in list_events:
                if event_type == 'add':
                    articles[title]['num-add'] += 1
                    articles[title]['add-articles'].add(to_article)
                else:
                    articles[title]['num-del'] += 1
                    articles[title]['del-articles'].add(to_article)
    for item in articles:
        articles[item]['add-articles'] = len(articles[item]['add-articles'])
        articles[item]['del-articles'] = len(articles[item]['del-articles'])
    print(f_path, len(articles))
    return articles


def process_count_add_del_events(
        num_cpus=50, file_type='pages-meta-history'):
    list_files = get_list_files_from(file_type=file_type)
    list_files = [os_basename(_).split('pages-meta-')[1].split('.7z')[0]
                  for _ in list_files]
    para_space = [os_join(r_path, 'internal_links', _ + '.json')
                  for _ in list_files]
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(func=count_add_del_events, iterable=para_space)
    pool.close()
    pool.join()
    f_write = open(os_join(r_path, f"stats-edge-events.json"), 'w')
    for articles in results_pool:
        for article in articles:
            json.dump([article, articles[article]], f_write)
            f_write.write('\n')
    f_write.close()


def draw_count_add_del_events():
    import matplotlib.pyplot as plt
    fr = open(os_join(r_path, f"stats-edge-events.json"), 'rb')
    x, y = [], []
    for each_line in fr:
        article, article_dict = json.loads(each_line.rstrip())
        if np.random.rand() < 0.02:
            a = article_dict['del-articles']
            b = article_dict['add-articles']
            if a != 0 and b != 0:
                x.append(a)
                y.append(b)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
    ax.scatter(x, y, marker='+', color='g', linewidth=.2, alpha=.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Added links', fontsize=16)
    ax.set_xlabel('Deleted links', fontsize=16)
    ax.set_title('Link events statistics', fontsize=16)
    fig.savefig('stat-link-to-articles.pdf',
                dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()


def draw_count_revisions():
    import matplotlib.pyplot as plt
    fr = open(os_join(r_path, f"stats-edge-events.json"), 'rb')
    x, y = dict(), dict()
    for each_line in fr:
        article, article_dict = json.loads(each_line.rstrip())
        if np.random.rand() <= 1.0:
            a = article_dict['num-revisions']
            b = article_dict['num-nontrivial-revisions']
            if a not in x:
                x[a] = 0
            x[a] += 1
            if b not in y:
                y[b] = 0
            y[b] += 1
    xx, yy = [], []
    for key in y:
        if key != 0 and y[key] != 0:
            xx.append(key)
            yy.append(y[key])
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    fig = plt.figure(figsize=(6.5, 6))
    ax = plt.gca()
    ax.scatter(xx, yy, marker='+', color='g', linewidth=.2, alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Power of Nontrivial Revisions', fontsize=14)
    ax.set_xlabel('Nontrivial Revisions', fontsize=14)
    ax.set_title('Nontrivial Revisions', fontsize=14)
    fig.savefig('stat-nontrivial-revisions.pdf',
                dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')


def process_dynamic_graph(
        num_cpus=60, file_type='pages-meta-history'):
    """
    This is for whole dynamic graph (include both add and del events).
    :param num_cpus:
    :param file_type:
    :return:
    """
    list_files = get_list_files_from(file_type=file_type)
    list_files = [os_basename(_).split('pages-meta-')[1].split('.7z')[0]
                  for _ in list_files]
    para_space = [(os_join(r_path, 'internal_links', _ + '.json'))
                  for _ in list_files]
    pool = multiprocessing.Pool(processes=num_cpus)
    results_pool = pool.map(func=get_all_edges, iterable=para_space)
    pool.close()
    pool.join()
    all_edge_list = []
    for articles in results_pool:
        all_edge_list.extend(articles)
    f_write = open(os_join(r_path, f"{dump_tag}-all-edges-tmp.json"), 'w')
    for _ in sorted(all_edge_list):
        json.dump(_, f_write)
        f_write.write('\n')
    f_write.close()

    nodes_id_map = dict()
    # the nodes remain unchanged.
    f_node_path = os_join(r_path, f"{dump_tag}-add-only-nodes-tmp.json")
    f_label_path = os_join(r_path, f"{dump_tag}-all-sorted-labels.json")
    f_write = open(os_join(r_path, f"en-wiki-nodes.json"), 'w')
    with open(f_node_path, 'rb') as f_node:
        with open(f_label_path, 'rb') as f_label:
            for ind, (l_node, l_label) in enumerate(zip(f_node, f_label)):
                created_at1, title1 = json.loads(l_node.rstrip())
                created_at2, title2, label = json.loads(l_label.rstrip())
                assert created_at1 == created_at2
                assert title1 == title2
                nodes_id_map[title1] = ind
                json.dump([ind, title1, created_at1, label], f_write)
                f_write.write(f"\n")
    f_write.close()
    ns0_titles, ns0_created_at = get_all_ns0_articles()
    f_edge_path = os_join(r_path, f"{dump_tag}-all-edges-tmp.json")
    f_write = open(os_join(r_path, f"en-wiki-edges.json"), 'w')
    with open(f_edge_path, 'rb') as f_edge:
        for l_edge in f_edge:
            timestamp, uu, vv, event_type = json.loads(l_edge.rstrip())
            while uu not in nodes_id_map:  # redirects could be a chain
                uu = ns0_titles[uu]
            while vv not in nodes_id_map:
                vv = ns0_titles[vv]
            edge_rec = [nodes_id_map[uu], nodes_id_map[vv],
                        timestamp, event_type]
            json.dump(edge_rec, f_write)
            f_write.write(f"\n")
    f_write.close()


def generate_nodes_labels():
    f_name = os_join(r_path, f"{dump_tag}-all-sorted-nodes.json")
    nodes = dict()
    with open(f_name) as f:
        for ind, each_line in enumerate(f):
            items = json.loads(each_line.rstrip())
            nodes[items[1]] = [ind, items[0]]
    ns0_titles, ns0_created_at = get_all_ns0_articles()
    root_path = "/your/path/data/wikipedia/dbpedia-ontology"
    labels = sorted([_ for _ in os.listdir(root_path) if _.endswith('json')])
    matched, unmatched = 0, 0
    for label in labels:
        label = label[:-5]
        f_path = root_path + f"/{label}.json"
        print(f"processing {f_path}")
        for item in json.load(open(f_path))['instances']:
            for key, val in item.items():
                article_title = key.split('/')[-1]
                if article_title not in ns0_titles:
                    print(article_title)
                    unmatched += 1
                    continue
                # has been redirected to a real title
                if ns0_titles[article_title] is not None:
                    real_title = ns0_titles[article_title]
                else:
                    real_title = article_title
                if real_title in nodes:
                    nodes[real_title].append(label)
                    matched += 1
                else:
                    unmatched += 1
                    # It should not happen!
                    print(f'Warning: unknown error, '
                          f'for title: {article_title}')
        ratio = matched / (matched + unmatched)
        print(f"{label} matched: {matched} unmatched: "
              f"{unmatched} ratio: {ratio:.4f}")
    list_all_nodes = [[]] * len(nodes)
    for title in nodes:
        items = nodes[title]
        list_all_nodes[items[0]].append(items[1])
        list_all_nodes[items[0]].append(title)
        if len(items) > 2:
            list_all_nodes[items[0]].extend(items[2:])
    f = open(os_join(r_path, f"{dump_tag}-all-sorted-labels.json"), 'w')
    for _ in list_all_nodes:
        json.dump(_, f)
        f.write('\n')
    f.close()


def get_real_articles():
    f_name = os_join(r_path, f"{dump_tag}-all-sorted-nodes.json")
    nodes = dict()
    with open(f_name) as f:
        for ind, each_line in enumerate(f):
            items = json.loads(each_line.rstrip())
            nodes[items[1]] = [ind, items[0]]
    return nodes


def get_label_articles():
    f_name = os_join(r_path, f"{dump_tag}-all-sorted-labels.json")
    nodes = dict()
    with open(f_name) as f:
        for ind, each_line in enumerate(f):
            timestamp, title, label = json.loads(each_line.rstrip())
            if label is not None:
                nodes[title] = label
    print(f"{len(nodes)} nodes have labels.")
    return nodes


def data_generation(rand_seed, training_rate, graph_path, draw_power):
    """Data generation for train and test. """
    import matplotlib.pyplot as plt
    rand = np.random.RandomState(seed=rand_seed)
    list_edges, graph, nodes, = [], dict(), dict()
    g = nx.Graph()
    with open(graph_path) as f:
        for each_line in f:
            uu, vv = each_line.rstrip().split(' ')
            uu, vv = int(uu), int(vv)
            nodes[uu] = ''
            nodes[vv] = ''
            list_edges.append((uu, vv))
            g.add_edge(uu, vv)
            if uu not in graph:
                graph[uu] = dict()
            graph[uu][vv] = ''
            if vv not in graph:
                graph[vv] = dict()
            graph[vv][uu] = ''
    mini_tr_edges = dict()
    for (uu, vv) in nx.minimum_spanning_edges(g):
        mini_tr_edges[(uu, vv)] = ''
        mini_tr_edges[(vv, uu)] = ''
    tr_nodes, tr_edges = dict(), []
    te_nodes, te_edges, te_labels = dict(), [], []
    for (uu, vv) in list_edges:
        if (uu, vv) in mini_tr_edges or rand.random() <= training_rate:
            tr_edges.append((uu, vv))
            tr_nodes[uu] = ''
            tr_nodes[vv] = ''
        else:
            te_edges.append((uu, vv))
            te_labels.append(1)
            te_nodes[uu] = ''
            te_nodes[vv] = ''
    index, num_nega = 0, len(te_edges)
    # sampling negative edges
    nega_edges = dict()
    while True:
        uu = rand.randint(0, len(nodes))
        vv = rand.randint(0, len(nodes))
        # sampling without repeats.
        if (uu, vv) in nega_edges:
            continue
        if vv not in graph[uu]:
            te_edges.append((uu, vv))
            te_labels.append(-1)
            te_nodes[uu] = ''
            te_nodes[vv] = ''
            nega_edges[(uu, vv)] = ''
            nega_edges[(vv, uu)] = ''
            index += 1
            if index == num_nega:
                break
    print(len(nodes), len(tr_nodes))
    assert len(nodes) == len(tr_nodes)
    if draw_power:
        graph = nx.from_edgelist(list_edges)
        print(f"# nodes: {nx.number_of_nodes(graph)}", end=" ")
        print(f"# edges: {nx.number_of_edges(graph)}", end=" ")
        print(f"# train edges: {len(tr_edges)}", end=" ")
        print(f"# test edges: {len(te_edges)}", end=" ")
        print(f"# cc: {nx.number_connected_components(graph)}")
        degrees = np.zeros(len(nodes))
        degree_freq = dict()
        for ind, degree in nx.degree(graph):
            degrees[ind] = degree
            if degree not in degree_freq:
                degree_freq[degree] = 0
            degree_freq[degree] += 1
        f_name = f'/your/path/data/wikipedia/enwiki-20200901/figs/' \
                 f'edges-{len(nodes)}-power-low.pdf'
        x, y = [], []
        for _ in sorted(degree_freq.keys()):
            x.append(_)
            y.append(degree_freq[_])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(x=x, y=y, marker='x')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02,
                    format='pdf')
        plt.close()
    return len(nodes), tr_edges, te_edges, te_labels


def get_statistics(f_path="/data/enwiki-20200901/"
                          "history2.xml-p30304p31181.json"):
    import matplotlib.pyplot as plt
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    stats = dict()
    stats['articles'] = set()
    stats['edit_gap'] = []
    stats['num_add_events'] = 0
    stats['num_del_events'] = 0
    stats['num_repeats'] = 0
    stats['add_links'] = set()
    stats['del_links'] = set()
    with open(f_path) as f:
        for ind, each_line in enumerate(f):
            items = json.loads(each_line.rstrip())
            page_title, _, revision_date, _, user_text, edge_events = items
            if ind == 0:
                pre_date = revision_date
            if page_title not in stats['articles']:
                stats['articles'].add(page_title)
            else:
                xx = date_to_int(revision_date) - date_to_int(pre_date)
                hours = (xx % 1000000) // 10000
                stats['edit_gap'].append(
                    (datetime.strptime(revision_date, date_format)
                     - datetime.strptime(pre_date,
                                         date_format)).days * 24 + hours + 1)
            pre_date = revision_date
            xx = [ii[1] for ii in edge_events if ii[0] == "add"]
            for _ in xx:
                stats['add_links'].add(_)
            stats['num_add_events'] += len(xx)
            stats['num_repeats'] += len(xx) - len(set(xx))
            xx = [ii[1] for ii in edge_events if ii[0] == "del"]
            for _ in xx:
                stats['del_links'].add(_)
            stats['num_del_events'] += len(xx)
            stats['num_repeats'] += len(xx) - len(set(xx))
    stats['intersect_links'] = \
        stats['del_links'].symmetric_difference(stats['add_links'])
    groups = dict()
    for _ in stats['edit_gap']:
        if _ not in groups:
            groups[_] = 0
        groups[_] += 1
    x, y = [], []
    for key, val in sorted(
            groups.items(), key=lambda xxx: xxx[1], reverse=True):
        x.append(key)
        y.append(val)
    plt.loglog(x, y, 'ro')
    plt.ylabel('Number of edits')
    plt.xlabel('Edit gap (hours)')
    plt.show()
    return stats


def download_pages_meta_history():
    # uncomment these lines if you want to reproduce the data
    get_file_from_url(f_base_name="dumpstatus.json")
    # download pages-meta-history
    get_all_ns0_titles()
    pool = multiprocessing.Pool(processes=3)
    f_list = [f_path for f_path in get_list_files_from("pages-meta-history")]
    pool.map(func=download_page_meta_history_files, iterable=f_list)
    pool.close()
    pool.join()


def get_list_labels_ordered():
    """ https://databus.dbpedia.org/dbpedia/collections/latest-core """
    soup = BeautifulSoup(open(
        os_join(r_path, "dbpedia-ontology", "index.html")), 'html.parser')
    all_labels_ordered = []
    for ul in soup.find_all('ul', recursive=True):
        for li_tag in ul.find_all('li', recursive=True):
            if len(li_tag.text) > 1:
                all_labels_ordered.append(
                    li_tag.text.lstrip().rstrip().split(' ')[0])
    all_labels_ordered = all_labels_ordered[1:779]
    general_labels = {'Place': 0, 'Person': 0, 'Work': 0, 'Species': 0,
                      'Organisation': 0, 'Food': 0, 'Event': 0, 'Device': 0,
                      'TimePeriod': 0, 'MeanOfTransportation': 0,
                      'PersonFunction': 0}
    valid_label_dict = dict()
    index1 = all_labels_ordered.index('Place')
    index2 = all_labels_ordered.index('WorldHeritageSite')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'Place'

    index1 = all_labels_ordered.index('Person')
    index2 = all_labels_ordered.index('SongWriter')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'Person'

    index1 = all_labels_ordered.index('Work')
    index2 = all_labels_ordered.index('foaf:Document')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'Work'

    index1 = all_labels_ordered.index('Species')
    index2 = all_labels_ordered.index('Moss')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'Species'

    index1 = all_labels_ordered.index('Organisation')
    index2 = all_labels_ordered.index('TradeUnion')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'Organisation'

    index1 = all_labels_ordered.index('Food')
    index2 = all_labels_ordered.index('Cheese')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'Food'

    index1 = all_labels_ordered.index('Event')
    index2 = all_labels_ordered.index('WrestlingEvent')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'Event'

    index1 = all_labels_ordered.index('Device')
    index2 = all_labels_ordered.index('Weapon')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'Device'

    index1 = all_labels_ordered.index('TimePeriod')
    index2 = all_labels_ordered.index('YearInSpaceflight')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'TimePeriod'

    index1 = all_labels_ordered.index('MeanOfTransportation')
    index2 = all_labels_ordered.index('Tram')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'MeanOfTransportation'

    index1 = all_labels_ordered.index('PersonFunction')
    index2 = all_labels_ordered.index('Profession')
    for _ in all_labels_ordered[index1:index2 + 1]:
        valid_label_dict[_] = 'PersonFunction'

    for label in general_labels:
        print(label)
    for label in valid_label_dict:
        print(label)
    return general_labels, valid_label_dict


def generate_nodes_categories():
    """ It can be downloaded from:
        https://databus.dbpedia.org/dbpedia/collections/latest-core
        Search: DBpedia Ontology instance types
    """
    f_path = os_join(r_path, "dbpedia-ontology",
                     "instance-types_lang=en_specific.ttl")
    general_labels, valid_label_dict = get_list_labels_ordered()
    all_articles, all_related_articles, unrelated = dict(), 0, dict()
    articles_labels = dict()
    with open(f_path, encoding="utf-8") as f:
        for ind, each_line in enumerate(f):
            each_line = each_line[:-4]
            article_title = each_line.split('> <')[0].split('/')[-1]
            category = each_line.split('> <')[2].split('/')[-1]
            all_articles[article_title] = ''
            if category in valid_label_dict:
                general_labels[valid_label_dict[category]] += 1
                all_related_articles += 1
                articles_labels[article_title] = valid_label_dict[category]
            else:
                if category not in unrelated:
                    unrelated[category] = 0
                unrelated[category] += 1
            if ind % 100000 == 0:
                print(ind, all_related_articles, len(all_articles))
    print(ind, all_related_articles, len(all_articles))
    for _ in general_labels:
        print(_, general_labels[_])
    for _ in sorted(unrelated.items(), key=lambda x: x[1], reverse=True):
        print(_)
    print(len(articles_labels))
    f_name = os_join(r_path, f"{dump_tag}-add-only-nodes-tmp.json")
    f_w = open(os_join(r_path, f"{dump_tag}-all-sorted-labels.json"), 'w')
    index1 = 0
    index2 = 0
    with open(f_name) as f:
        for ind, each_line in enumerate(f):
            date, title = json.loads(each_line.rstrip())
            if title not in all_articles:
                index1 += 1
            label = None
            if title in articles_labels:
                index2 += 1
                if index2 % 10000 == 0:
                    print(index1, index2, title)
                label = articles_labels[title]
            json.dump([date, title, label], f_w)
            f_w.write('\n')
    f_w.close()


def main():
    # process_articles_sketch(60)
    # process_articles_created_at(55)  # step 1
    # process_get_article_internal_links()  # step 2
    # process_add_only_dynamic_graph()  # step 3
    # generate_nodes_categories()  # step 4
    # prepare_add_only_nodes()
    # prepare_add_only_edges()
    # build_add_only_graph()
    # process_count_add_del_events()
    # process_dynamic_graph()
    # get_statistics()
    # plot_event_gaps()
    # fix_redirects()
    nodes = dict()
    with open(os_join(r_path, f"en-wiki-nodes.json")) as f_node:
        for ind, each_line in enumerate(f_node):
            node_id, title, created_at, label = json.loads(each_line)
            nodes[node_id] = ''
    g = nx.Graph()
    with open(os_join(r_path, f"en-wiki-edges.json")) as f_node:
        for ind, each_line in enumerate(f_node):
            uu, vv, timestamp, event = json.loads(each_line)
            g.add_edge(uu, vv)
            if ind % 1000000 == 0:
                print(ind)
    print(len(nodes), len(g.nodes))
    largest_cc = None
    for ind, cc in enumerate(nx.connected_components(g)):
        print(ind, len(cc))
        if largest_cc is None or len(largest_cc) < len(cc):
            largest_cc = cc
    re_index = dict()
    for ind, node in enumerate(sorted(largest_cc)):
        re_index[node] = ind
    # only 8 nodes will be removed.
    # 3391693 Protowenella 20110125214715
    # 3391700 Tichkaella 20110125215212
    # 3665123 World_Faith 20111123212934
    # 3671374 Frank_Fredericks_(musician) 20111130215025
    # 4679438 VMB2 20150212102031
    # 4679450 MDC08 20150212105251
    # 6137393 Dora_Mmari_Msechu 20200829034719
    # 6137403 List_of_ambassadors_of_Tanzania_to_the_
    # Nordic_Countries,_Baltic_States_and_Ukraine 20200829041315
    fw = open("/your/path/data/kdd21/enwiki-20/enwiki_nodes.json", 'w')
    with open(os_join(r_path, f"en-wiki-nodes.json")) as f_node:
        for ind, each_line in enumerate(f_node):
            node_id, title, created_at, label = json.loads(each_line)
            if node_id not in re_index:
                print(node_id, title, created_at)
                continue
            json.dump([re_index[node_id], title, created_at, label], fw)
            fw.write("\n")
    fw.close()
    fw = open("/your/path/data/kdd21/enwiki-20/enwiki_edges.json", 'w')
    with open(os_join(r_path, f"en-wiki-edges.json")) as f_node:
        for ind, each_line in enumerate(f_node):
            uu, vv, timestamp, event = json.loads(each_line)
            if uu not in re_index or vv not in re_index:
                print(uu, vv, timestamp)
                continue
            json.dump([re_index[uu], re_index[vv], timestamp, event], fw)
            fw.write("\n")
    fw.close()


def fix_redirects():
    r_path = "/your/path/data/kdd21/enwiki20/"
    articles = dict()
    with open(os_join(r_path, "enwiki20_nodes.json")) as f:
        for each_line in f:
            node_id, title, timestamp, _ = json.loads(each_line)
            articles[node_id] = [title, timestamp]
    counts = 0
    all_edges = []
    for ind, each_line in enumerate(
            open(os_join(r_path, "enwiki20_edges.json"))):
        uu, vv, timestamp, event = json.loads(each_line)
        # This is due to some articles are redirected to a new page
        # The old page was deleted.
        # In total, there are about 0.3365% percentage of such edges.
        if articles[vv][1] > timestamp:
            counts += 1
            # we update the edge timestamp to the node creation time
            all_edges.append([uu, vv, articles[vv][1], event])
        else:
            all_edges.append([uu, vv, timestamp, event])
        if ind % 1000000 == 0:
            print(ind // 1000000, counts, counts / (ind + 1))
    fw = open(os_join(r_path, "enwiki20_edges-fixed.json"), 'w')
    for item in sorted(all_edges, key=lambda x: x[2]):
        json.dump(item, fw)
        fw.write("\n")
    fw.close()


if __name__ == '__main__':
    main()

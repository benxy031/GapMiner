/**
 * Stratum protocol implementation for GapMiner
 *
 * Copyright (C)  2014  The Gapcoin developers  <info@gapcoin.org>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <jansson.h>
#include <cerrno>
#include <cstring>

#ifndef WINDOWS
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "Stratum.h"
#include "utils.h"

#include "Opts.h"

using namespace std;

/**
 * seconds to wait until trying to reconnect
 */
#define RECONNECT_TIME 15

/* synchronization mutexes */
pthread_mutex_t Stratum::creation_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t Stratum::connect_mutex  = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t Stratum::send_mutex     = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t Stratum::shares_mutex   = PTHREAD_MUTEX_INITIALIZER;

/* the socket of this */
int Stratum::tcp_socket = 0;

/* the server address */
string *Stratum::host = NULL;

/* the server port */
string *Stratum::port = NULL;

/* the user */
string *Stratum::user = NULL;

/* the users password */
string *Stratum::password = NULL;

/* the mining shift */
uint16_t Stratum::shift = 0;

/* the only instance of this */
Stratum *Stratum::only_instance;

/* indicates that this is running */
bool Stratum::running = true;

/**
 * returns the position of the first new line character in the given string
 */
static inline ssize_t new_line_pos(char *str, size_t len) {
  
  size_t i;
  for (i = 0; i < len && str[i] != '\n'; i++);

  return i;
}

/**
 * reallocs new space if needed
 */
static inline ssize_t check_memory(char **buffer, ssize_t *capacity, ssize_t wanted) {

  if (*capacity < 1024) {
    *capacity = 1024;
    char *ptr = (char *) realloc(*buffer, *capacity);

    if (ptr == NULL)
      return -1;

    *buffer = ptr;
  }
  
  while (*capacity < wanted) {
    *capacity *= 2;
    
    char *ptr = (char *) realloc(*buffer, *capacity);

    if (ptr == NULL)
      return -1;

   *buffer = ptr;
  }

  return 0;
}

/**
 * receives on line form a given socket file descriptor
 * Note: it reallocs space if needed
 * NOTE: *buffer has to be an dynamic allocated address
 */
ssize_t recv_line(int sock_fd, char **buffer, ssize_t *capacity, int flags) {
  
  static char recv_buff[1024];
  static ssize_t recved_size = 0;

  ssize_t size = 0, size_one = 0;

  for (;;) {
    
    /* malloc new space if needed */
    if (check_memory(buffer, capacity, size + 1023) == -1)
      return -1;

    /* receive on portion form sender or use old received portion */
    if (recved_size > 0) {
      
      memmove(*buffer, recv_buff, recved_size);
      size_one = recved_size;
      recved_size = 0;
    } else
      size_one = recv(sock_fd, (*buffer) + size, 1023, flags);

    /* check for errors */
    if (size_one == -1) return -1;

    /* search for a new line char */
    ssize_t new_line_char = new_line_pos((*buffer) + size, size_one);

    /* adjust size */
    new_line_char += size;
    size += size_one;

    /* we found a new line char */
    if (size > new_line_char) {
      
      /* part of the next line */
      recved_size = size - (new_line_char + 1);

      /* save part of the next line for future calls */
      if (recved_size > 0)
        memmove(recv_buff, (*buffer) + new_line_char + 1, recved_size);

      (*buffer)[new_line_char] = 0x0;

      size = new_line_char;
      break;
    }
  } 

  return size;
}

/* access or create the only instance of this */
Stratum *Stratum::get_instance(string *host, 
                               string *port, 
                               string *user,
                               string *password,
                               uint16_t shift,
                               Miner *miner) {
  
  log_str("get_instance", LOG_D);
  pthread_mutex_lock(&creation_mutex);

  /* allow only one creation */
  if (host  != NULL &&
      port  != NULL && 
      miner != NULL && 
      user  != NULL && 
      password != NULL && 
      shift >= 14 &&
      only_instance == NULL) {

    Stratum::host = host;
    Stratum::port = port;
    Stratum::user = user;
    Stratum::password = password;
    Stratum::shift    = shift;

#ifdef WINDOWS
    WSADATA data;
    int res = WSAStartup(MAKEWORD(2,2), &data);
    if (res != 0) {
      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "Failed to initialize winsocket" << endl;
      pthread_mutex_unlock(&io_mutex);
    }
#endif

    
    reconnect();

    only_instance = new Stratum(miner);
  }

  pthread_mutex_unlock(&creation_mutex);

  return only_instance;
}

/**
 * (re)start an keep alive tcp connection
 */
void Stratum::reinit() {

  log_str("reinit", LOG_D);
  int ret = -1;

  do {
    if (running && tcp_socket > 0) {
      close(tcp_socket);
      tcp_socket = -1;
    }

    if (running) {
      
      tcp_socket = socket(AF_INET, SOCK_STREAM, 0); 
      
      /* socket successfully created ? */
      if (running && tcp_socket >= 0) {
      
        /* set keep alive option */
        int optval = 1;
        ret = setsockopt(tcp_socket,                                                 
                         SOL_SOCKET,                                                 
                         SO_KEEPALIVE,                                               
                         (const char *) &optval,                                                    
                         sizeof(int));
      } 
      
      /* recreate on error */
      if (running && (tcp_socket < 0 || ret < 0)) {
      
        pthread_mutex_lock(&io_mutex);
        cout << get_time() << "failed to create tcp socket: ";
        cout << strerror(ret) << endl;
        cout << "retrying after " << RECONNECT_TIME << " seconds..." << endl;
        pthread_mutex_unlock(&io_mutex);
        sleep(RECONNECT_TIME);
      }
    }
  } while (running && (tcp_socket < 0 || ret < 0));
}

/**
 * (re)connect to a given addr
 */
bool Stratum::connect_to_addr(struct addrinfo *addr) {

  log_str("connect_to_addr", LOG_D);
  bool ret = false;

  if (running) {
    
    int cret = -1;

    /* (re)create socket */
    reinit();
    
    if (running) {
      
      cret = connect(tcp_socket, addr->ai_addr, addr->ai_addrlen);
      
      /* wait and retry on failure */
      if (!cret)
        ret = true;
    }
  }

  return ret;
}

/**
 * (re)connect to a given pool
 */
void Stratum::reconnect() {

  log_str("reconnect", LOG_D);

  /* only one thread are allowed to (re)connect */
  pthread_mutex_lock(&connect_mutex);

  struct addrinfo hints;
  struct addrinfo *result, *rp;

  do {

    /* (re)create socket */
    reinit();

    if (running) {
      memset(&hints, 0, sizeof(struct addrinfo));
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_STREAM;
     
      int ret = getaddrinfo(host->c_str(), port->c_str(), &hints, &result);
     
      if (ret != 0) {
        pthread_mutex_lock(&io_mutex);
        cout << get_time() << "failed to obtain pool ip: " << strerror(ret) << endl;
        pthread_mutex_unlock(&io_mutex);
      }
     
      for (rp = result; rp != NULL; rp = rp->ai_next)
        if (connect_to_addr(rp))
          break;
     
      if (rp == NULL) {               /* No address succeeded */
        pthread_mutex_lock(&io_mutex);
        cout << get_time() << "failed to connect to pool" << endl;
        cout << "retrying after " << RECONNECT_TIME << " seconds..." << endl;
        pthread_mutex_unlock(&io_mutex);
        sleep(RECONNECT_TIME);
      }
    }
  } while (running && rp == NULL);
  

  pthread_mutex_unlock(&connect_mutex);
}

/* creates a new Stratum instance */
Stratum::Stratum(Miner *miner) {

  log_str("create", LOG_D);
  this->n_msgs = 0;
  this->targs = new ThreadArgs(miner, &shares);
  pthread_create(&thread, NULL, recv_thread, targs);

  getwork();
}

Stratum::~Stratum() {
  
  log_str("delete", LOG_D);
  targs->running = false;
  pthread_join(thread, NULL);
  
  delete targs;
}

Stratum::ThreadArgs::ThreadArgs(Miner *miner, map<int, double> *shares) {
  this->miner   = miner;
  this->shares  = shares;
  this->running = true;
}

/**
 * Thread that listens for new messages from the server.
 * it updates miners, and prints share information
 *
 * Messages that can be received by this:
 *
 * Block notification:
 *   "{ "id": null, "method": "blockchain.block.new", "params": 
 *      { "data": <block hex data to solve>, 
 *      "difficulty": <target difficulty> } }"
 *
 *   This message informs the clients that a new block arrived and they should
 *   update their mining work.
 *
 * Getwork response:
 *   "{ "id": <id of the request>, "result": { "data": <block data to solve>,    
 *      "difficulty": <target difficulty> }, "error": <null or errors string> }"
 *
 *   This message send the current block data when the client asks for a
 *   work update.
 *
 * Sendwork response:
 *   "{ "id": <id of the share>, "result": <true/false>,
 *      "error": <null or errors string>}"
 *
 *   This message sends the result for a submitted share, which can be
 *   accepted (true) ore stale (false)
 * 
 * Note: "error" is currently ignored.
 */
void *Stratum::recv_thread(void *arg) {
  
  log_str("recv_thread started", LOG_D);
  ThreadArgs *targs = (ThreadArgs *) arg;
  Miner *miner = targs->miner;
  map<int, double> *shares = targs->shares;
  ssize_t buf_len = 1024;
  char *buffer = (char *) malloc(sizeof(char) * buf_len);


  if (Opts::get_instance()->has_extra_vb()) {
    extra_verbose_log(get_time() + "Stratum thread started");
  }

  while (targs->running) {
    
    /* receive message from server */
    if (recv_line(tcp_socket, &buffer, &buf_len, 0) < 0) {

      log_str("recv: \"" + string(buffer, buf_len) + "\"", LOG_D);
      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "Error receiving message form server: " << endl;
      pthread_mutex_unlock(&io_mutex);

      /* reset connection */
      reconnect();
      get_instance()->getwork();
    }

    json_t *root, *real_root;
    json_error_t error;

    if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
      extra_verbose_log(get_time() + " stratum-response: " + string(buffer));
    }

    root = json_loads(buffer, 0, &error);
    real_root = root;
 
    if(!root) {
      log_str("jansson error: on line " + itoa(error.line) + ":" + error.text, LOG_W);
      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "jansson error: on line " << error.line;
      cout << ": " << error.text << endl;
      pthread_mutex_unlock(&io_mutex);
      continue;
    }
 
    if (!json_is_object(root)) {
      log_str("can not parse server response", LOG_W);
      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "can not parse server response" << endl;
      pthread_mutex_unlock(&io_mutex);
      json_decref(real_root);
      continue;
    }
    json_t *j_id = json_object_get(root, "id");
 
    /* parse response */
    if (json_is_integer(j_id)) {
      int id = json_number_value(j_id);

      json_t *result = json_object_get(root, "result");

      /* share response */
      if (json_is_boolean(result)) {
        process_share(shares, id, json_is_true(result));

      /* getwork response */
      } else if (json_is_object(result)) {
        parse_block_work(miner, result);

      } else if (json_is_null(result)) {
        log_str("Found share stale!", LOG_I);
        pthread_mutex_lock(&io_mutex);
        cout << get_time() << "Found share stale!" << endl;
        pthread_mutex_unlock(&io_mutex);

      } else {
        log_str("can not parse server response", LOG_W);
        pthread_mutex_lock(&io_mutex);
        cout << get_time() << "can not parse server response" << endl;
        pthread_mutex_unlock(&io_mutex);
      }
      json_decref(real_root);

    /* block notify message */
    } else {
      json_t *params = json_object_get(root, "params");
      
      if (json_is_object(params)) {
        parse_block_work(miner, params);

      } else {
        log_str("can not parse server response", LOG_W);
        pthread_mutex_lock(&io_mutex);
        cout << get_time() << "can not parse server response" << endl;
        pthread_mutex_unlock(&io_mutex);
      }
      json_decref(real_root);
    }
  }

  log_str("recv_thread stopped", LOG_D);
  return NULL;
}

/* helper function which processes an response share */
void Stratum::process_share(map<int, double> *shares, int id, bool accepted) {
  
  log_str("process_share", LOG_D);
  pthread_mutex_lock(&shares_mutex);
  bool share_not_found = (shares->find(id) == shares->end());
  pthread_mutex_unlock(&shares_mutex);

  if (share_not_found) {
    log_str("Received invalid server response", LOG_W);
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "Received invalid server response" << endl;
    pthread_mutex_unlock(&io_mutex);
    return;
  }
  

  log_str("Found Share: " + itoa(shares->at(id) * TWO_POW48) + " => " +
      (accepted ? "accepted" : "stale!"), LOG_I);

  pthread_mutex_lock(&io_mutex);
  cout.precision(7);
  cout << get_time();
  cout << "Found Share: " << fixed << shares->at(id);
  cout << "  =>  " <<  (accepted ? "accepted" : "stale!");
  cout << endl;
  pthread_mutex_unlock(&io_mutex);

  pthread_mutex_lock(&shares_mutex);
  shares->erase(id);
  pthread_mutex_unlock(&shares_mutex);
}

/* helper function to parse a json block work in the form of:
 * "{ "data": <block data to solve>, "difficulty": <target difficulty> }"
 */
void Stratum::parse_block_work(Miner *miner, json_t *result) {

  log_str("parse_block_work", LOG_D);
  json_t *tdiff;

  tdiff  = json_object_get(result, "difficulty");
  result = json_object_get(result, "data");

  /* parse difficulty */
  if (!json_is_integer(tdiff)) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "can not parse server difficulty" << endl;
    pthread_mutex_unlock(&io_mutex);
    return;
  }

  uint64_t nDiff = json_number_value(tdiff);

  /* parse block data */
  if (!json_is_string(result)) {
    log_str("can not parse server difficulty", LOG_W);
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "can not parse server block data" << endl;
    pthread_mutex_unlock(&io_mutex);
    return;
  }
  string data = json_string_value(result);
  
  BlockHeader head(&data);
  head.target = nDiff;
  head.shift  = shift;

  /* update work */
  if (miner->started())
    miner->update_header(&head);
  else
    miner->start(&head);

  log_str("Got new target: " + itoa(head.target) + " @ " + 
      itoa(head.difficulty), LOG_I);

  pthread_mutex_lock(&io_mutex);
  cout.precision(7);
  cout << get_time() << "Got new target: ";
  cout << fixed << (((double) head.target) / TWO_POW48) << " @ ";
  cout << fixed << (((double) head.difficulty) / TWO_POW48) << endl;
  pthread_mutex_unlock(&io_mutex);
}

/**
 * sends a given BlockHeader to the server 
 * with a stratum request, the response should
 * tell if the share was accepted or not.
 *
 * The format should be:
 *   "{ "id": <id of the share>, "result": <true/false>,
 *      "error": <null or errors string> }"
 */
bool Stratum::sendwork(BlockHeader *header) {

  stringstream ss;
  ss << "{\"id\": " << n_msgs;
  ss << ", \"method\": \"mining.submit\", \"params\": ";
  ss << "[ \"" << *user << "\", \"" << *password;
  ss << "\", \"" << header->get_hex()  << "\" ] }\n";

  bool error;
  string share = ss.str();
  if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
    extra_verbose_log(get_time() + " stratum-payload: " + share);
  }
  do {
    error = false;

    /* send the share to the pool */
    log_str("sendwork: \"" + share + "\"", LOG_D);
    size_t ret = send(tcp_socket, share.c_str(), share.length(), 0);
 
    if (ret != share.length()) {
 
      log_str("Submitting share failed", LOG_W);
      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "Submitting share failed" << endl;
      pthread_mutex_unlock(&io_mutex);
      error = true;
      reconnect();
    }

  } while (running && error);

  pthread_mutex_lock(&shares_mutex);
  shares[n_msgs] = ((double) header->get_pow().difficulty()) / TWO_POW48;
  n_msgs++;
  pthread_mutex_unlock(&shares_mutex);

  return true;
}

/**
 * request new work, return is always NULL, 
 * because it handles response internally .
 *
 * The stratum response for this request should be: 
 *   "{ "id": <id of the request>, "result": 
 *      { "data": <block hex data to solve>,    
 *        "difficulty": <target difficulty> }, 
 *        "error": <null or errors string> }"
 */
BlockHeader *Stratum::getwork() {
  
  stringstream ss;
  ss << "{\"id\": " << n_msgs;
  ss << ", \"method\": \"mining.request\", \"params\": ";
  /* not optimal password should be hashed */
  ss << "[ \"" << *user << "\", \"" << *password  << "\" ] }\n";

  string request = ss.str();

  bool error;
  do {
    error = false;

    log_str("getwork: \"" + request + "\"", LOG_D);
    size_t ret = send(tcp_socket, request.c_str(), request.length(), 0);
 
    if (ret != request.length()) {
 
      log_str("Requesting work failed", LOG_W);
      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "Requesting work failed" << endl;
      pthread_mutex_unlock(&io_mutex);
      error = true;
      reconnect();
    }

  } while (running && error);

  pthread_mutex_lock(&shares_mutex);
  n_msgs++;
  pthread_mutex_unlock(&shares_mutex);
  return NULL;
}

void Stratum::stop() {

  log_str("stop", LOG_D);
  Stratum::running = false;
  close(tcp_socket);
}

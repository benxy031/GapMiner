/**
 * Implementation of some utility functions
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
#ifndef __STDC_FORMAT_MACROS 
#define __STDC_FORMAT_MACROS 
#endif
#ifndef __STDC_LIMIT_MACROS  
#define __STDC_LIMIT_MACROS  
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <gmp.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "utils.h"

/**
 * converts a given integer to a string
 */
string itoa(uint64_t i) {

  char a[32];

  if (i == 0) {
    a[0] = '0';
    a[1] = '\0';
    return string(a);
  }

  uint64_t j;
  for (j = 0; i != 0; j++, i /= 10)
    a[j] = 48 + (i % 10);

  a[j] = '\0';
  string str = string(a);;
  return string(str.rbegin(), str.rend());
}

/* returns the current time */
string get_time() {
  time_t timer;
  char buffer[25];
  struct tm* tm_info;

  time(&timer);
  tm_info = localtime(&timer);

  strftime(buffer, 25, "\r[%Y-%m-%d %H:%M:%S] ", tm_info);
  return string(buffer);
}

/**
 * converts a given double to a string
 */
string dtoa(double d, unsigned precision) {
  
  if (precision == 0)
    return itoa((uint64_t) d);

  return itoa((uint64_t) d) + "." +
         itoa((uint64_t) (d * pow(10, precision) - 
              ((uint64_t) d) * pow(10, precision)));
}

/**
 * mutex to avoid mutual exclusion by writing output
 */
pthread_mutex_t io_mutex = PTHREAD_MUTEX_INITIALIZER;

/* log file descriptor */
static int log_fd = 0;

static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

static int extra_log_fd = -1;
static pthread_mutex_t extra_log_mutex = PTHREAD_MUTEX_INITIALIZER;
static off_t extra_log_bytes = 0;
static const off_t EXTRA_LOG_MAX_BYTES = 10LL * 1024LL * 1024LL * 1024LL;

#define write_str(fd, str) write(fd, str, strlen(str))

#ifndef NO_LOGGING
/* logs the given string */
void log_string(string str, int status) {
  
  pthread_mutex_lock(&log_mutex);
  if (log_fd == 0) {
    log_fd = open("gapminer.log", O_CREAT|O_TRUNC|O_WRONLY, 00777);
    if (log_fd < 0) {
      perror("failed to open log");
      return;
    }
  }

  if (status == LOG_E)
    write_str(log_fd, "[EE]");
  else if (status == LOG_W)
    write_str(log_fd, "[WW]");
  else if (status == LOG_I)
    write_str(log_fd, "[II]");
  else 
    write_str(log_fd, "[DD]");

  string time = get_time();
  write(log_fd, time.c_str() + 1, time.length() - 2);

  for (unsigned i = 0; i < str.length(); i++)
    if (str[i] == '\n')
      str[i] = ' ';

  write_str(log_fd, str.c_str());
  write_str(log_fd, "\n");

  pthread_mutex_unlock(&log_mutex);
}
#endif

void extra_verbose_log(const string &str) {
  pthread_mutex_lock(&extra_log_mutex);

  if (extra_log_fd == -1) {
    extra_log_fd = open("tests", O_CREAT | O_TRUNC | O_WRONLY, 00777);
    if (extra_log_fd < 0) {
      perror("failed to open extra verbose log");
      pthread_mutex_unlock(&extra_log_mutex);
      return;
    }
    extra_log_bytes = 0;
  }

  string msg = str;
  if (msg.empty() || msg[msg.length() - 1] != '\n')
    msg.push_back('\n');

  const size_t msg_len = msg.length();
  if (extra_log_bytes + (off_t) msg_len > EXTRA_LOG_MAX_BYTES) {
    if (ftruncate(extra_log_fd, 0) == 0)
      lseek(extra_log_fd, 0, SEEK_SET);
    extra_log_bytes = 0;
  }

  ssize_t written = write(extra_log_fd, msg.c_str(), msg_len);
  if (written > 0)
    extra_log_bytes += written;

  pthread_mutex_unlock(&extra_log_mutex);
}

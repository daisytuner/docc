#include "sdfg/util/utils_curl.h"
#include <iostream>

HttpResult post_json(CURL* curl, const std::string& url, const std::string& payload, struct curl_slist* headers) {
    HttpResult result;
    std::string response_data;

    curl_easy_setopt(
        curl,
        CURLOPT_WRITEFUNCTION,
        +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
            auto* resp = static_cast<std::string*>(userdata);
            resp->append(ptr, size * nmemb);
            return size * nmemb;
        }
    );
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());

    if (headers) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    result.curl_code = curl_easy_perform(curl);
    result.body = response_data;

    if (result.curl_code != CURLE_OK) {
        result.error_message = curl_easy_strerror(result.curl_code);
        return result;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &result.http_status);

    if (result.http_status == 401) {
        result.error_message = "[ERROR] RPC optimization query authentication issue: " +
                               std::to_string(result.http_status) + ", body: " + result.body;
    } else if (result.http_status < 200 || result.http_status >= 204) {
        result.error_message = "HTTP error: " + std::to_string(result.http_status) + ", body: " + result.body;
    }

    return result;
}

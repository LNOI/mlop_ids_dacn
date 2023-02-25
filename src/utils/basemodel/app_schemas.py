from typing import Optional
from pydantic import BaseModel


class IDSSchema(BaseModel):
    duration : list[int]
    protocol_type : list[int]
    flag : list[int]
    src_bytes : list[int]
    dst_bytes : list[int]
    land : list[int]
    wrong_fragment : list[int]
    urgent : list[int]
    hot : list[int]
    num_failed_logins : list[int]
    logged_in : list[int]
    num_compromised : list[int]
    root_shell : list[int]
    su_attempted : list[int]
    num_file_creations : list[int]
    num_shells : list[int]
    num_access_files : list[int]
    is_guest_login : list[int]
    count : list[int]
    srv_count : list[int]
    serror_rate : list[float]
    rerror_rate : list[float]
    same_srv_rate : list[float]
    diff_srv_rate : list[float]
    srv_diff_host_rate : list[float]
    dst_host_count : list[int]
    dst_host_srv_count : list[int]
    dst_host_diff_srv_rate : list[float]
    dst_host_same_src_port_rate : list[float]
    dst_host_srv_diff_host_rate : list[float]
    
class Datasetchema(BaseModel):
    pass

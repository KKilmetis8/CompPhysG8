t_max  = len(m_array)
t_diff = t_max - t
term1  = (1/t_diff) * sum(m_array[:t_diff] * m_array[t:])
term2  = (1/t_diff**2) * sum(m_array[:t_diff]) * sum(m_array[t:])
chi    = term1 - term2
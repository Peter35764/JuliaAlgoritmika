function search_sorted_last(sorted::AbstractVector, value)::Int
    (isempty(sorted) || value < sorted[1]) && return 0
    i_beg = 0
    i_end = length(sorted)
    while i_beg < i_end
        i_mean = Int64(ceil(Int(i_beg + i_end)/2))
        if sorted[i_mean] <= value
            if i_mean == length(sorted) || value < sorted[i_mean+1]
                return i_mean
            end
            i_beg = i_mean
        else
            i_end = i_mean
        end
    end
end

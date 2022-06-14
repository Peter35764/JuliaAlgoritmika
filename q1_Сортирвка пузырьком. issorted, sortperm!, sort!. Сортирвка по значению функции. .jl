# 1)	Сортировка пузырьком. Issorted, sortperm!, sort!. Сортировка по значению функции .
# n = parse(Int, readline())
function bubblesort!(massive::AbstractArray)
    n = length(massive)
    for i in 1:n-1
        istranspose = false
        for i in firstindex(massive):lastindex(massive) - i - 1
            if massive[i] > massive[i + 1]
                massive[i], massive[i + 1] = massive[i + 1], massive[i]
                istranspose = true
            end
        end
        if istranspose == false
            break
        end
    end
    return massive
end

a = [4, 5, 2, 6]
bubblesort!(a)

function iissorted(massive::AbstractArray)
    for i in firstindex(massive):lastindex(massive) - 1
        if massive[i] > massive[i+1]
            return false
        end
    end
    return true
end

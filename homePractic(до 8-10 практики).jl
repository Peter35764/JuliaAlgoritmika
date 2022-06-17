using Plots
## Практика 1

# Длина последовательности

function length_(A)
    len = 0
    for _ in A
        len += 1
    end
    return len
end

# Сумма членов последовательности

function sum_(A)
    s = eltype(A)(0)
    for a in A
        s += a
    end
    return s
end

# Произведение членов последовательности

function prod_(A)
    p = eltype(A)(1)
    for a in A
        p *= a
    end
    return p
end

# Максимальное значение членов последовательности

function maximum_(A)
    M = typemin(eltype(A)) # m = -Inf   #Наименьшее значение, которое может быть представлено данным (действительным) числовым типом данных T
    for a in A
        M = max(M,a)
    end
    return M
end

# Минимальное значение членв последовательности

function maximum_(A)
    m = typemin(eltype(A)) # m = -Inf
    for a in A
        m = min(m,a)
    end
    return m
end

# Индекс максимального значения членов последовательности

function argmax_(A)
    @assert !isempty(A)
    imax = firstindex(A)
    for k in eachindex(A)
        if A[k] > A[imax] 
            imax = k
        end
    end
    return imax
end

# Значение многочлена в точке, вычисленное по последовательности его коэффициентов, заданной по убыванию степеней, по хеме Горнера


function evalpoly_(x,A)
    Q = first(A) # - это есть a_0
    for a in @view A[2:end]
        Q=Q*x+a
    end
    return Q
end

# Сортировка числового массива вставками
# Сложность алгортма сортировки вставками оценивается как O(n^2).
function insertsort!(A)
    n=length(A)
    for k in eachindex(A) #2:n
        # часть массива A[1:k-1] уже отсортирована
        op_insert!(A,k)
    end
    return A
end

op_insert!(A,k) =
    while k>1 && A[k-1] > A[k]
        A[k-1], A[k] = A[k], A[k-1]
        k -= 1
    end

#practic1_a = rand(1:100, 15)
#println("Генерируем массив случайных чисел: ", practic1_a)
#println("Сортировка вставками: ", insertsort!(practic1_a))

# Сортировка подсчётом
#=
function calcsort!(a; values = maximum(a))
    massive = zeros(Int64, a)
    return massive
end
=#

#----------------------------------------------------------------------------------------------

## Практика 2

# Быстрая осртировка Хоара 
# Алгоритмическая сложность алгортма Хоара оцениваеся как O(Nlog(N))

function quick_sort!(A)
    if isempty(A)
        return A
    end
    N = length(A)
    K, M = part_sort!(A, A[rand(1:N)]) 
    return A
end

function part_sort!(A, b)
    N = length(A)
    K,L,M=0,0,N

    @inbounds while L < M
        if A[L+1] == b
            L += 1
        elseif  A[L+1] > b
            A[L+1],A[M] = A[M], A[L+1]
            M -= 1
        else 
            L +=1; K +=1
            A[L], A[K] = A[K], A[L]
        end
    end
    return K, M+1
end

#practic2_a = rand(1:100, 15)
#println("Генерируем массив случайных чисел: ", practic2_a)
#println("Быстрая сортировка Хоара: ", quick_sort!(practic2_a))

# Сортировка пузырьком

function bubblesort!(a)          # O(n^2)
    n = length(a)
    for k in 1:n-1
        istranspose = false
        for i in firstindex(a):lastindex(a)-k
            if a[i]>a[i+1]
                a[i], a[i+1] = a[i+1], a[i]
                istranspose = true
            end
        end
        if istranspose == false
            break
        end
    end
    return a
end


#Вычисление k-й порядковой статистики

function order_statistiks!(A::AbstractVector{T}, i::Integer)::T where T
    function find(index_range)
        left_range, right_range = part_sort!(A, index_range, A[rand(index_range)])
        if i in left_range
            return find(left_range)
        elseif i in right_range
            return find(right_range)
        else
            return A[i]
        end
    end
    find(firstindex(A):lastindex(A))
end

# Медиана массива 

function medianmassive(massive)
    N = length(massive)
    if N % 2 == 1
        return order_statistiks!(massive, Int((N+1)/2))
    else
        return (order_statistiks!(massive, Int(N/2)) + order_statistiks!(massive, Int((N/2)+1)))/2
    end
end

# Сортировка расчёсыванием 

function combsort!(a; factor = 1.2473309)  # Параметр factor = 1.2473309 наиболее эффективен в данном алгоритме
    step = length(a)
    while step >= 1
        for i in 1:length(a)-step
            if a[i] > a[i + step]
                a[i], a[i + step] = a[i + step], a[i]
            end
        end
        step = Int(floor(step/factor))

    end
    bubblesort!(a)
end

# Сортировка Шелла
# Алгоритмическая сложность O(n^2)
function shellSort(A)
    inp=A
    n=length(inp)
    h = n / 2
    h=round(h)
    h=(Int64)(h)
    while h>0
        for i in h:n
            t = inp[i]
            j = i
            while j > h && inp[j - h] > t 
                inp[j] = inp[j - h]
                j -= h
            end
            inp[j] = t
        end
        h = h / 2
        h=round(h)
        h=(Int64)(h)
    end
    return inp
end

#practic2_1_a = rand(1:100, 15)
#println("Генерируем массив случайных чисел: ", practic2_1_a)
#println("Сортировка Шелла: ", shellSort(practic2_1_a))



#----------------------------------------------------------------------------------------------

# Практика 3 

# Такую модификацию "пузырьковой" сортировки назвают сортировкой "перемешиванием", или "шейкерной" сортировкой (Двунаправленная пузырьковая сортировка)

function mixersort!(a)
    i_beg = 1
    i_end = length(a)
    while i_beg < i_end
        @inbounds for i in i_beg:i_end-1 # макрос @inbounds отменяет контроль выхода за пределы массива
            if a[i] > a[i+1]
                a[i], a[i+1] = a[i+1], a[i]
            end
        end
        i_end -= 1
        @inbounds for i in i_end:-1:i_beg+1 # i меняется в сторону уменьшения (в диапазоне шаг отрицательный)
            if a[i-1] > a[i]
                a[i-1], a[i] = a[i], a[i-1]
            end
        end
        i_beg += 1
    end
    return a
end

# Алгоритмическая сложность данной сортировки O(n^2)

#practic3_a = rand(1:100, 15)
#println("Генерируем массив случайных чисел: ", practic3_a)
#println("Двунаправленная пузырьковая сортировка: ", mixersort!(practic3_a))

# Быстрое O(N) вычисление первых (аналогично, - последних)  порядковых статистик 
#=
function minimums(array, k)
    N = length(array)
    k_minimums = sort(array[1:k])
    i = k
    # ИНВАРИАНТ: issorted(k_mins) && k_mins - содержит k наименьших элементов в array[1:i]
    while i < length(array)
        i += 1
        if array[i] < k_minimums[end]
            k_minimums[end] = array[i]
            insert_end!(k_minimums)
        end
    end
    return k_minimums
end            

function insert_end!(array)::Nothing
    j = length(array)
    while j>1 && array[j-1] > array[j]
        array[j-1], array[j] = array[j], array[j-1]
        j -= 1
    end
end
=#
# Сортировка слиянием 

# Алгоритмическая сложность сортировки слиянием O(nlog(n))

function Base.merge!(a1, a2, a3)::Nothing
    i1, i2, i3 = 1, 1, 1
    @inbounds while i1 <= length(a1) && i2 <= length(a2) # @inbounds - передотвращает проверки выхода за пределы массивов
        if a1[i1] < a2[i2]
            a3[i3] = a1[i1]
            i1 += 1
        else
            a3[i3] = a2[i2]
            i2 += 1
        end
        i3 += 1
    end
    if i1 > length(a1)
        a3[i3:end] .= @view(a2[i2:end]) # Если бы тут было: a3[i3:end] = @view(a2[i2:end]), то это привело бы к лишним аллокациям (к созданию промежуточного массива)
    else
        a3[i3:end] .= @view(a1[i1:end])
    end
    nothing
end

function mergesort!(a)
    b = similar(a) # - вспомогательный массив того же размера и типа, что и массив a
    N = length(a)
    n = 1 # n - текущая длина блоков
    while n < N
        K = div(N,2n) # - число имеющихся пар блоков длины n
        for k in 0:K-1
            merge!(@view(a[(1:n).+k*2n]), @view(a[(n+1:2n).+k*2n]), @view(b[(1:2n).+k*2n]))
        end
        if N - K*2n > n # - осталось еще смержить блок длины n и более короткий остаток
            merge!(@view(a[(1:n).+K*2n]), @view(a[K*2n+n+1:end]), @view(b[K*2n+1:end]))
        elseif 0 < N - K*2n <= n # - оставшуюся короткую часть мержить несчем
            b[K*2n+1:end] .= a[K*2n+1:end]
        end
        a, b = b, a
        n *= 2
    end
    if isodd(Int(log2(n))) # - если цикл был выполнен нечетное число раз
        b[:] .= a[:]  # a = copy(b) - это было бы не то же самое, т.к. тут получилась бы ссылка на новый массив, который создаст функция copy
        a = b
    end
    return a # - ссылка на исходный массив (проверить, что это так, можно с помощью ===)
end

#practic3_1_a = rand(1:100, 15)
#println("Генерируем массив случайных чисел: ", practic3_1_a)
#println("Сортировка слиянием: ", mergesort!(practic3_1_a))

#быстрое возведение в степень без рекурсии

function fast_step(x,y)
    k, t, p = y, 1, x

#ИНВАРИАНТ: p^k*t=a^n 
    while k>0
        if (k%2==0) # k - четное
            k ÷= 2
            p *= p # т.к. p^k = (p*p)^k*(t/2)
        else
            k -= 1
            t *= p # т.к. p^k * t = p^(k-1)*(p*t)
        end   
    end 
    return t
end

#Быстрое возведение в степень с рекурсией

function binpow(x,y)
    if (y==0) 
        return 1
    elseif (y%2==0) 
        return binpow(x*x,y/2)
    else 
        return x*binpow(x,y-1)
    end
end

# Решение нелинейного уравнения методом деления отрезка пополам
# Практика 5 

function bisect(f::Function, a, b, ε)
    y_a=f(a)
    # ИНВАРИАНТ: f(a)*f(b) < 0 (т.е. (a,b) - содержит корень)
    while b-a > ε
        x_m = (a+b)/2
        y_m=f(x_m)
        if y_m==0
            return x_m
        end
        if y_m*y_a > 0 
            a=x_m
        else
            b=x_m
        end
    end
    return (a+b)/2
end

# Алгоритм Евклида

function NOD(m, n)
    a, b = m, n
    #ИНВАРИАНТ: НОД(a,b)==НОД(m,n)
    while b != 0
        a, b = b, a % b # a % b - целочисленный остаток от деления a на b
    end
    return a
end


# Расширенный алгоритм Евклида

# m, n - заданные целые
function Advanced_Euclid_Algorithm(m,n)
    a, b = m, n
    u_a, v_a = 1, 0
    u_b, v_b = 0, 1
#=
ИНВАРИАНТ: 
    НОД(m,n)==НОД(a,b)
    a = u_a*m + v_a*n 
    b = u_b*m + v_b*n
=#

    while b != 0
        k = a÷b
        a, b = b, a % b 
    #УТВ: a % b = a-k*b - остаток от деления a на b
        u, v = u_a, v_a
        u_a, v_a = u_b, u_a
        u_b, v_b = u-k*u_b, v-k*v_b
    end
    return (u_a+n)%n
end


function reverse_ring(A)
    n=length(A)
    Q=A
    for i in 2:n
        if (NOD(A[i],A[n])==1)
            Q[i]=Advanced_Euclid_Algorithm(A[i],A[end])
        end
    end
    return Q
end

#A=[1,3,7]
#println(reverse_ring(A))

# Алгоритмы по лекции 4 ( Polinomial) практики: 6, 7, 8
# Пользовательский тип "многочлены"

struct Polynomial{T} <:Integer
    coeff::Vector{T} 
    function Polynomial{T}(coeff) where T
        n=0
        for c in reverse(coeff)
            if c==0
                n+=1
            end
        end
        new(coeff[1 : end - n]) 
    end
end
# p = Polynomial{Int}([1,2,3.0]) - example of using constructor
deg(p::Polynomial) = length(p.coeff) - 1
#=
# Переопределние операций +,-,*
function Base. +(p::Polynomial{T}, q::Polynomial{T})::Polynomial{T} where T
    len_p, len_q = length(p.coeff) , length(q.coeff)
    if  len_p >= len_q 
        coeff = similar(p.coeff)
        coeff[1:len_q] .= (@view(p.coeff[1:len_q]) .+ q) 
    else
        coeff = similar(q.coeff)
        coeff[1:len_p] .= (p .+ @view(q.coeff[1:len_p]))
    end
    i, n = lastindex(coeff), 0
    while i > 0 && coeff[i] == 0
        n += 1
        i -= 1
    end
    resize!(coeff, length(coeff)-n)
    return Polynomial{T}(coeff)
end

function Base. -(p::Polynomial{T}, q::Polynomial{T})::Polynomial{T} where T
    len_p, len_q = length(p.coeff), length(q.coeff)
    if len_p > len_q
        coeff = similar(p.coeff)
        coeff[1:len_q] .= ( @view(p.coeff[1:len_q]) .- q )
    else
        coeff = similar(q.coeff)
        coeff[1:len_p] .= (p .- @view(q.coeff[1:len_p]))
    end 

    i, n = lastindex(coeff), 0
    while i > 0 && coeff[i] == 0
        n += 1
        i -= 1
    end
    resize!(coeff, length(coeff)-n)
    return Polynomial{T}(coeff)
end

function Base. *(p::Polynomial{T}, q::Polynomial{T})::Polynomial{T} where T
    coeff = zeros(T, deg(p) + deg(q)+1)
    for i in eachindex(p.coeff), j in eachindex(q.coeff)
            coeff[i+j - 1] += p.coeff[i]*q.coeff[j]
    end
    i,n = lastindex(coeff),0
    while i > 0 && coeff[i] == 0
        n+=1
        i-=1
    end
    resize(coeff, length(coeff) - n)
    return Polynomial{T}(coeff)
end
=#
function display(polinom::Polynomial{T}) where T # стандартный вывод многочлена
    s = 'a'
    n = length(polinom.coeff)
    print(s)
    s+=1
    for i in 2 : n
        print(" + ",s ,"^",polinom.coeff[i])
        s+=1
    end
    println()
    return 
end

function polyval(m::Polynomial{Int64}, x::Any) # Значение многочлена в точке

    mass = m.coeff
    rezult = 0
    for i in 1:length(mass)
        rezult += x^mass[i] 
    end
    return rezult
end

# Пользовательский тип кольцо вычетов

struct Residue{T, M}
    value::T                                   # %
    Residue{T,M}(value) where {T,M} = new(value % M) 
end

a = Residue{Int,5}(7)
b = Residue{Int,6}(7)

Base. +(a::Residue{T,M}, b::Residue{T,M}) where{T,M} = (a.value + b.value) ÷ M
Base. *(a::Residue{T,M}, b::Residue{T,M}) where{T,M} = (a.value * b.value) ÷ M

Base. /(a::Residue{T,M}, b::Residue{T,M}) where{T,M} = a * inv(b)
#=
Base. -(a::Reidue{T,M})where{T,M} = Residue{T,M}(M - a.value)

Base. -(a::Residue{T,M}, b::Residue{T,M}) = a + (-b)


(==)(a::Residue,b::Residue) where{T,M} 
    while a>0
        a += M
    end
    while b > 0
        b +=M
    end
    return (a % M) == (b % M)
end
=#

# Практика 6


function rx(x::Float64)
    return (-(cos(x)-x)/(-sin(x)-1))
end


#=
function main()
    n = parse(Int, readline())

    if n == 1??
        newton(x -> (cos(x)-x)/(sin(x)+1), 0.5)
    end
end
=#

function eyler(n)
    sum_n = 0
    a_n = 1
    for k in 1:n + 1
        sum_n += a_n
        a_n /= k
    end
    return sum_n
end

function eyler(x)
    #if x < 0
    #    x = abs(x)
    #end
    sum_n = 0
    a_n = 1
    k = 1
    while sum_n + a_n != sum_n
        sum_n += a_n
        a_n /= k
        k += 1
    end
    #if 
    #    sum_n = 1/sum_n
    #end
    println(k)
    return sum_n
end

# ряд Маклорена для e^x

function sinus(x, ε)
    xx = x^2
    a = x
    k = 1
    s = typeof(x)(0) # - преобразование к 0 нужного типа, что обеспечит стабильность типа переменной s
    while abs(a) > ε
        s += a
        a = -a*xx/2k/(2k+1)
        k += 1
    end
    return s
end

function newton(r::Function, x; epsilon = 1e-8, max_num_iter = 20)
    num_iter = 0
    r_x = r(x)
    while num_iter < max_num_iter && abs(r_x) > epsilon
        x += r_x
        r_x = r(x)
        num_iter += 1
    end

    if abs(r_x) <= epsilon
        return x
    else
        return NaN
    end
end

function ln_x(x)
    sum_n = 0
    a_n = x - 1
    k = 1
    while sum_n + a_n != sum_n
        sum_n += a_n/k
        a_n = -a_n*(x - 1)
        k += 1
    end
    #println(k)
    return sum_n
        #=
        (-1)^(k + 1)*() 
    =#
end

function Root(x)# корень

    sum_n = 0
    a_n = 1
    k = 1
    while sum_n + a_n != sum_n
        sum_n += a_n
        a_n = ((-1) * (2k - 1) *x * a_n)/2k
       
        #a_n = (-1)/(2k+2)*2k*(2k + 1) *x*a_n
        k += 1
        
    end
    return sum_n
end

function punkt3(x) #-
    sum_n = 0
    u_n = 1
    v_n = 1
    w_n = 1
    k = 1
    while sum_n + u_n*(v_n + w_n) != sum_n
        
        u_n = -u_n*x^2
        v_n = v_n * ( x + 1 )
        w_n = w_n/(4x^2+2x)
        sum_n += u_n*(v_n + w_n)
        k += 1
    end
    return sum_n
end

p = [1,2,3] 
function proizvodn(p)
    h=[]
    for i in 1:length(p)-1
        push!(h,p[i]*(length(p)-i))
    end
    return h
end


function rx_mnogochlen(x,p)
    return -(eval_poly(x,p))/(eval_poly(x,proizvodn(p)))
end
function eval_poly(x,A) # значение многочлена в точке 
    Q = first(A) # - это есть a_0
    for a in @view A[2:end]
        Q=Q*x+a
    end
    return Q
end
#С помощью функции newton и написанной ранее функции polyval, возвращающей значение многочлена (заданного вектором коэффициентов) 
# и его производной в заданной точке, написать функцию, возвращающее приближенное значение комплексого корня многочлена, ближайшего к заданной точке комплексой плоскости.

function NewTon_complex(x, eps, max_num_iter,p)
    num_iter=0
    r_x=rx_mnogochlen(x,p)
    while num_iter < max_num_iter && abs(r_x) > eps
        x+=r_x
        r_x=rx_mnogochlen(x,p)
        num_iter +=1
    end
    if abs(r_x) < eps
        return x
    else
        return NaN
    end
end

z=NewTon_complex(-1+1im,1e-8,20,p)
println(conj(z))

#plot(scatter([1,z,conj(z)])) #выводит точку на графике 



